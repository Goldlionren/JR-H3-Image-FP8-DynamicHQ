import logging
from contextvars import ContextVar
from pathlib import Path

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if Path(__file__).with_name("ENABLE_DYNAMIC_FP8.txt").exists():
    from comfy.quant_ops import TensorCoreFP8E4M3Layout
    from comfy.ldm.minimax.model import MiniMaxH3ImageModel

    _active = ContextVar("jr_h3_fp8_active", default=False)
    _stats = ContextVar("jr_h3_fp8_stats", default=None)
    _logged = False

    if not getattr(TensorCoreFP8E4M3Layout,
                   "_jr_dynamic_installed", False):

        _original_quantize = TensorCoreFP8E4M3Layout.quantize

        @classmethod
        def _dynamic_quantize(cls, tensor, scale=None, **kwargs):
            if _active.get() and scale is None:
                scale = "recalculate"
                stats = _stats.get()
                if stats is not None:
                    stats["calls"] += 1

            return _original_quantize(
                tensor, scale=scale, **kwargs
            )

        TensorCoreFP8E4M3Layout.quantize = _dynamic_quantize
        TensorCoreFP8E4M3Layout._jr_dynamic_installed = True

        _original_forward = MiniMaxH3ImageModel.forward

        def _h3_forward(self, *args, **kwargs):
            global _logged

            stats = {"calls": 0}
            token_a = _active.set(True)
            token_s = _stats.set(stats)

            try:
                return _original_forward(self, *args, **kwargs)
            finally:
                _stats.reset(token_s)
                _active.reset(token_a)

                if not _logged:
                    logging.warning(
                        "[JR-H3-FP8] Dynamic FP8 inputs: %d",
                        stats["calls"]
                    )
                    _logged = True

        MiniMaxH3ImageModel.forward = _h3_forward

        logging.warning(
            "[JR-H3-FP8] Dynamic activation scaling ENABLED"
        )
