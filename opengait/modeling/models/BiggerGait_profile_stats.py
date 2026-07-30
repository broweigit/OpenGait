import json
import os

import torch

from .BiggerGait_DINOv2 import BiggerGait__DINOv2
from .BiggerGait_DINOv2_Projection_Mask_based import BiggerGait__DINOv2__Projection_Mask_Based
from .BiggerGait_SAM_3D_Body import BiggerGait__SAM3DBody_Gaitbase_Share
from .BiggerGait_SAM_3D_Body_official import BiggerGait__SAM3DBody_Official_Gaitbase_Share
from .BiggerGait_SAM_3D_Body_projection_mask import (
    BiggerGait__SAM3DBody__Projection_Mask_Gaitbase_Share,
)
from .BiggerGait_SAM_3D_Body_projection_mask_OT_based_SparseTopK4 import (
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_Gaitbase_Share,
)
from .BiggerGait_SAM_3D_Body_projection_mask_direct_relocation import (
    BiggerGait__SAM3DBody__Projection_Mask_DirectRelocation_Gaitbase_Share,
)
from .BiggerGait_SAM_3D_Body_learned_mask import (
    BiggerGait__SAM3DBody__Learned_Mask_Gaitbase_Share,
)


class _ProfileStatsMixin:
    def build_network(self, model_cfg):
        self.profile_enabled = bool(model_cfg.get("profile_enabled", True))
        self.profile_case_name = model_cfg.get("profile_case_name", self.__class__.__name__)
        self.profile_upstream_name = model_cfg.get("profile_upstream_name", "unknown")
        self.profile_downstream_name = model_cfg.get("profile_downstream_name", "unknown")
        self.profile_method_name = model_cfg.get("profile_method_name", self.__class__.__name__)
        self.profile_dummy_batch = int(model_cfg.get("profile_dummy_batch", 1))
        self.profile_dummy_seq = int(model_cfg.get("profile_dummy_seq", 1))
        self.profile_runtime_seq = int(model_cfg.get("profile_runtime_seq", 30))
        self.profile_runtime_warmup = int(model_cfg.get("profile_runtime_warmup", 3))
        self.profile_runtime_repeats = int(model_cfg.get("profile_runtime_repeats", 10))
        self.profile_runtime_amp = bool(model_cfg.get("profile_runtime_amp", True))
        self.profile_runtime_enabled = bool(model_cfg.get("profile_runtime_enabled", True))
        self.profile_dump_json = bool(model_cfg.get("profile_dump_json", True))
        self.profile_prune_modules = list(model_cfg.get("profile_prune_modules", []))
        self._profile_forward_active = False
        super().build_network(model_cfg)

    def init_parameters(self):
        preload_params_m, preload_trainable_params_m = self._collect_param_stats()
        super().init_parameters()
        for module_name in self.profile_prune_modules:
            if module_name in self._modules:
                setattr(self, module_name, torch.nn.Identity())
                self.msg_mgr.log_info(
                    f"[ProfileStats] Pruned inactive module: {module_name}"
                )
        if self.profile_enabled:
            self._run_profile_stats(
                preload_params_m=preload_params_m,
                preload_trainable_params_m=preload_trainable_params_m,
            )

    def forward(self, inputs):
        if self._profile_forward_active:
            return super().forward(inputs)
        labs = inputs[1] if isinstance(inputs, (list, tuple)) and len(inputs) > 1 else None
        return self._dummy_retval(labs)

    def _dummy_retval(self, labs):
        batch = int(labs.numel()) if torch.is_tensor(labs) else 1
        anchor = None
        for param in self.parameters():
            if param.requires_grad:
                anchor = param
                break
        if anchor is None:
            device = next(self.parameters()).device
            dummy = torch.zeros((), device=device, requires_grad=True)
        else:
            device = anchor.device
            dummy = anchor.sum() * 0.0
        return {
            "training_feat": {"profile_dummy_loss": dummy},
            "visual_summary": {},
            "inference_feat": {"embeddings": torch.zeros(batch, 1, device=device)},
        }

    def _build_profile_inputs(self, device, sequence_length=None):
        height, width = self.image_size * 2, self.image_size
        sequence_length = self.profile_dummy_seq if sequence_length is None else int(sequence_length)
        float_dtype = torch.float32
        for param in self.parameters():
            if torch.is_floating_point(param):
                float_dtype = param.dtype
                break
        rgb = torch.randn(
            (self.profile_dummy_batch, sequence_length, 3, height, width),
            device=device,
            dtype=float_dtype,
        )
        labels = torch.zeros(self.profile_dummy_batch, dtype=torch.long, device=device)
        aux = torch.ones(self.profile_dummy_batch, dtype=float_dtype, device=device)
        return (([rgb, aux], labels, None, None, None),)

    def _collect_param_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params / 1e6, trainable_params / 1e6

    def _collect_gait_topology(self):
        num_fpn_paths = None
        unique_gaitbase_modules = None

        gait_net = getattr(self, "Gait_Net", None)
        if gait_net is None:
            return num_fpn_paths, unique_gaitbase_modules

        if hasattr(gait_net, "Gait_List"):
            gait_list = list(gait_net.Gait_List)
            num_fpn_paths = len(gait_list)
            unique_gaitbase_modules = len({id(module) for module in gait_list})
            return num_fpn_paths, unique_gaitbase_modules

        return num_fpn_paths, unique_gaitbase_modules

    def _profile_flops_with_torch_profiler(self, dummy_inputs):
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        with torch.no_grad():
            with profile(activities=activities, with_flops=True, record_shapes=False, profile_memory=False) as prof:
                _ = self(dummy_inputs[0])

        total_flops = 0
        for event in prof.key_averages():
            event_flops = getattr(event, "flops", 0) or 0
            total_flops += int(event_flops)
        return float(total_flops / 1e9)

    def _profile_runtime_and_memory(self, device):
        if not self.profile_runtime_enabled:
            return {
                "runtime_sequence_ms": None,
                "runtime_frame_ms": None,
                "peak_memory_allocated_mib": None,
                "peak_memory_reserved_mib": None,
            }

        runtime_inputs = self._build_profile_inputs(
            device, sequence_length=self.profile_runtime_seq
        )
        was_training = self.training
        timings_ms = []
        self.eval()
        self._profile_forward_active = True
        try:
            use_cuda = device.type == "cuda"
            amp_enabled = bool(self.profile_runtime_amp and use_cuda)
            if use_cuda:
                torch.cuda.empty_cache()
            with torch.no_grad():
                for _ in range(self.profile_runtime_warmup):
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=amp_enabled,
                    ):
                        retval = self(runtime_inputs[0])
                    del retval
                if use_cuda:
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)

                for _ in range(self.profile_runtime_repeats):
                    if use_cuda:
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                    else:
                        import time
                        start_time = time.perf_counter()

                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=amp_enabled,
                    ):
                        retval = self(runtime_inputs[0])
                    del retval

                    if use_cuda:
                        end.record()
                        end.synchronize()
                        timings_ms.append(float(start.elapsed_time(end)))
                    else:
                        import time
                        timings_ms.append(
                            float((time.perf_counter() - start_time) * 1000.0)
                        )

            if not timings_ms:
                raise RuntimeError("profile_runtime_repeats must be positive.")
            sorted_timings = sorted(timings_ms)
            middle = len(sorted_timings) // 2
            if len(sorted_timings) % 2:
                median_batch_ms = sorted_timings[middle]
            else:
                median_batch_ms = (
                    sorted_timings[middle - 1] + sorted_timings[middle]
                ) / 2.0

            peak_allocated = None
            peak_reserved = None
            if use_cuda:
                peak_allocated = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
                peak_reserved = torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)
            sequence_count = max(self.profile_dummy_batch, 1)
            frame_count = max(
                self.profile_dummy_batch * self.profile_runtime_seq, 1
            )
            return {
                "runtime_sequence_ms": median_batch_ms / sequence_count,
                "runtime_frame_ms": median_batch_ms / frame_count,
                "peak_memory_allocated_mib": peak_allocated,
                "peak_memory_reserved_mib": peak_reserved,
            }
        finally:
            self._profile_forward_active = False
            self.train(was_training)

    def _run_profile_stats(self, preload_params_m=None, preload_trainable_params_m=None):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")

        total_params_m, trainable_params_m = self._collect_param_stats()
        frozen_params_m = total_params_m - trainable_params_m
        num_fpn_paths, unique_gaitbase_modules = self._collect_gait_topology()
        flops_g = None
        unsupported_ops = {}
        was_training = self.training
        detached_modules = {}
        dummy_inputs = None

        try:
            from fvcore.nn import FlopCountAnalysis

            self.eval()
            self._profile_forward_active = True
            self.to(device=device, dtype=torch.float32)
            for module_name in ("SAM_Engine",):
                module = self._modules.pop(module_name, None)
                if module is not None:
                    object.__setattr__(self, module_name, module)
                    detached_modules[module_name] = module
            dummy_inputs = self._build_profile_inputs(device)
            analysis = FlopCountAnalysis(self, dummy_inputs)
            if hasattr(analysis, "unsupported_ops_warnings"):
                analysis.unsupported_ops_warnings(False)
            if hasattr(analysis, "uncalled_modules_warnings"):
                analysis.uncalled_modules_warnings(False)
            if hasattr(analysis, "tracer_warnings"):
                analysis.tracer_warnings("none")
            flops_g = float(analysis.total() / 1e9)
            if hasattr(analysis, "unsupported_ops"):
                unsupported_ops = {
                    str(k): int(v) for k, v in analysis.unsupported_ops().items()
                }
        except Exception as exc:
            self.msg_mgr.log_warning(f"[ProfileStats] fvcore FLOPs profiling failed: {exc}")
            try:
                if dummy_inputs is None:
                    dummy_inputs = self._build_profile_inputs(device)
                flops_g = self._profile_flops_with_torch_profiler(dummy_inputs)
            except Exception as profiler_exc:
                self.msg_mgr.log_warning(f"[ProfileStats] torch.profiler FLOPs profiling failed: {profiler_exc}")
        finally:
            for module_name, module in detached_modules.items():
                object.__delattr__(self, module_name)
                self.add_module(module_name, module)
            self._profile_forward_active = False
            self.train(was_training)

        runtime_info = {
            "runtime_sequence_ms": None,
            "runtime_frame_ms": None,
            "peak_memory_allocated_mib": None,
            "peak_memory_reserved_mib": None,
        }
        try:
            runtime_info = self._profile_runtime_and_memory(device)
        except Exception as runtime_exc:
            self.msg_mgr.log_warning(
                f"[ProfileStats] runtime/memory profiling failed: {runtime_exc}"
            )

        profile_info = {
            "case_name": self.profile_case_name,
            "method_name": self.profile_method_name,
            "upstream": self.profile_upstream_name,
            "downstream": self.profile_downstream_name,
            "preload_params_m": None if preload_params_m is None else round(float(preload_params_m), 4),
            "preload_trainable_params_m": None if preload_trainable_params_m is None else round(float(preload_trainable_params_m), 4),
            "total_params_m": round(float(total_params_m), 4),
            "trainable_params_m": round(float(trainable_params_m), 4),
            "frozen_params_m": round(float(frozen_params_m), 4),
            "runtime_frame_ms": None if runtime_info["runtime_frame_ms"] is None else round(float(runtime_info["runtime_frame_ms"]), 4),
            "runtime_sequence_ms": None if runtime_info["runtime_sequence_ms"] is None else round(float(runtime_info["runtime_sequence_ms"]), 4),
            "peak_memory_allocated_mib": None if runtime_info["peak_memory_allocated_mib"] is None else round(float(runtime_info["peak_memory_allocated_mib"]), 4),
            "peak_memory_reserved_mib": None if runtime_info["peak_memory_reserved_mib"] is None else round(float(runtime_info["peak_memory_reserved_mib"]), 4),
            "runtime_protocol": {
                "batch_size": self.profile_dummy_batch,
                "sequence_frames": self.profile_runtime_seq,
                "warmup_repeats": self.profile_runtime_warmup,
                "timed_repeats": self.profile_runtime_repeats,
                "amp_fp16": self.profile_runtime_amp,
                "statistic": "median",
            },
            "flops_g": None if flops_g is None else round(float(flops_g), 4),
            "num_fpn_paths": num_fpn_paths,
            "unique_gaitbase_modules": unique_gaitbase_modules,
            "dummy_input_shape": [
                self.profile_dummy_batch,
                self.profile_dummy_seq,
                3,
                self.image_size * 2,
                self.image_size,
            ],
            "unsupported_ops": unsupported_ops,
        }

        self.msg_mgr.log_info(f"[ProfileStats] case={profile_info['case_name']}")
        self.msg_mgr.log_info(
            "[ProfileStats] upstream={} | downstream={} | preload={:.4f}M | #Params={:.4f}M | trainable={:.4f}M | FLOPs={}".format(
                profile_info["upstream"],
                profile_info["downstream"],
                0.0 if profile_info["preload_params_m"] is None else profile_info["preload_params_m"],
                profile_info["total_params_m"],
                profile_info["trainable_params_m"],
                "N/A" if profile_info["flops_g"] is None else f"{profile_info['flops_g']:.4f}G",
            )
        )
        self.msg_mgr.log_info(
            "[ProfileStats] frozen={:.4f}M | peak_allocated={} | "
            "runtime/frame={} | runtime/{}-frame-sequence={} | "
            "batch={} | precision={}".format(
                profile_info["frozen_params_m"],
                "N/A" if profile_info["peak_memory_allocated_mib"] is None
                else f'{profile_info["peak_memory_allocated_mib"]:.2f} MiB',
                "N/A" if profile_info["runtime_frame_ms"] is None
                else f'{profile_info["runtime_frame_ms"]:.4f} ms',
                self.profile_runtime_seq,
                "N/A" if profile_info["runtime_sequence_ms"] is None
                else f'{profile_info["runtime_sequence_ms"]:.4f} ms',
                self.profile_dummy_batch,
                "FP16 AMP" if self.profile_runtime_amp else "FP32",
            )
        )

        if num_fpn_paths is not None or unique_gaitbase_modules is not None:
            self.msg_mgr.log_info(
                f"[ProfileStats] gait_topology: num_fpn_paths={num_fpn_paths}, "
                f"unique_gaitbase_modules={unique_gaitbase_modules}"
            )
        if unsupported_ops:
            self.msg_mgr.log_info(f"[ProfileStats] unsupported_ops={unsupported_ops}")

        if rank == 0 and self.profile_dump_json:
            os.makedirs(self.save_path, exist_ok=True)
            path = os.path.join(self.save_path, "profile_stats.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(profile_info, handle, indent=2, ensure_ascii=False)
            self.msg_mgr.log_info(f"[ProfileStats] Saved stats to {path}")


class BiggerGait__DINOv2_ProjectionMask_ProfileStats(
    _ProfileStatsMixin, BiggerGait__DINOv2__Projection_Mask_Based
):
    """Profile the implemented DINOv2-G + offline projected-mask forward."""

    def _build_profile_inputs(self, device, sequence_length=None):
        sequence_length = (
            self.profile_dummy_seq
            if sequence_length is None
            else int(sequence_length)
        )
        height, width = self.image_size * 2, self.image_size
        rgb = torch.randn(
            (
                self.profile_dummy_batch,
                sequence_length,
                3,
                height,
                width,
            ),
            device=device,
            dtype=torch.float32,
        )
        labels = torch.zeros(
            self.profile_dummy_batch, dtype=torch.long, device=device
        )
        vertices = torch.zeros(6890, 3, device=device, dtype=torch.float32)
        cam_t = torch.tensor(
            [0.0, 0.0, 3.0], device=device, dtype=torch.float32
        )
        cam_int = torch.tensor(
            [
                [500.0, 0.0, width / 2.0],
                [0.0, 500.0, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=torch.float32,
        )
        frame = {
            "pred_vertices": vertices,
            "pred_cam_t": cam_t,
            "cam_int": cam_int,
        }
        sam_decoder = [
            [frame for _ in range(sequence_length)]
            for _ in range(self.profile_dummy_batch)
        ]
        return (([rgb, sam_decoder], labels, None, None, None),)


class BiggerGait__DINOv2_ProfileStats(_ProfileStatsMixin, BiggerGait__DINOv2):
    pass


class BiggerGait__SAM3DBody_ProfileStats_Gaitbase_Share(
    _ProfileStatsMixin, BiggerGait__SAM3DBody_Gaitbase_Share
):
    pass


class BiggerGait__SAM3DBody_ProjectionMask_ProfileStats_Gaitbase_Share(
    _ProfileStatsMixin,
    BiggerGait__SAM3DBody__Projection_Mask_Gaitbase_Share,
):
    """Profile A3: SAM 3D Body + projected-mask original branch only."""

    pass


class BiggerGait__SAM3DBody_LearnedMask_ProfileStats_Gaitbase_Share(
    _ProfileStatsMixin,
    BiggerGait__SAM3DBody__Learned_Mask_Gaitbase_Share,
):
    """Profile old-A3 downstream with encoder-only learned mask."""

    pass


class BiggerGait__SAM3DBody_Official_ProfileStats_Gaitbase_Share(
    _ProfileStatsMixin, BiggerGait__SAM3DBody_Official_Gaitbase_Share
):
    pass

class PuppetGait__SAM3DBody_SparseTopK4_ProfileStats_Gaitbase_Share(
    _ProfileStatsMixin,
    BiggerGait__SAM3DBody__Projection_Mask_OT_Based_SparseTopK4_Gaitbase_Share,
):
    pass

class PuppetGait__SAM3DBody_DirectRelocation_ProfileStats_Gaitbase_Share(
    _ProfileStatsMixin,
    BiggerGait__SAM3DBody__Projection_Mask_DirectRelocation_Gaitbase_Share,
):
    pass
