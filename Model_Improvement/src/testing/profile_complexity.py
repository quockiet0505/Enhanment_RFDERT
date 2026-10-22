"""Auditable forward-only complexity estimate; never silently accept missing ops."""
import math
from collections import Counter


def measure(core, size=640, device="cuda:0"):
    import torch
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn.jit_handles import get_shape

    def elements(value):
        shape = get_shape(value)
        if shape is None or any(n is None for n in shape):
            raise ValueError("Unknown tensor shape in FLOP trace")
        return math.prod(shape)

    def pointwise(name, cost=1):
        def handle(inputs, outputs):
            count = elements(outputs[0])
            complex_output = outputs[0].type().scalarType() in {"ComplexFloat", "ComplexDouble"}
            factor = 2 if complex_output else 1
            if complex_output and name == "mul":
                complex_inputs = sum(v.type().kind() == "TensorType" and
                    v.type().scalarType() in {"ComplexFloat", "ComplexDouble"} for v in inputs)
                factor = 6 if complex_inputs == 2 else 2
            if name == "abs":
                cost_abs = 4 if inputs[0].type().scalarType() in {"ComplexFloat", "ComplexDouble"} else 1
                return Counter({name: count * cost_abs})
            return Counter({name: count * cost * factor})
        return handle

    def fft(inputs, outputs):
        name = outputs[0].node().kind().split("::")[-1]
        # Conventional radix-2 equivalent estimate, not the actual cuFFT plan.
        # Real transforms: 2.5*N*log2(HW); complex: 5*N*log2(HW).
        inverse_real = name == "fft_irfft2"
        real = name in {"fft_rfft2", "fft_irfft2"}
        shape = get_shape(outputs[0] if inverse_real else inputs[0])
        h, w = shape[-2:]
        n = math.prod(shape)
        return Counter({name: (2.5 if real else 5) * n * math.log2(h*w)})

    def attention(inputs, outputs):
        q, k = get_shape(inputs[0]), get_shape(inputs[1])
        b = math.prod(q[:-2])
        l, s, d = q[-2], k[-2], q[-1]
        # Two matrix products (one FMA = one), scale plus softmax ~6/entry.
        return Counter({"attention": b*l*s*(2*d+6)})

    def avg_pool(inputs, outputs):
        kernel = inputs[1].toIValue()
        if kernel is None:
            kernel = [v.toIValue() for v in inputs[1].node().inputs()]
        return Counter({"avg_pool2d": elements(outputs[0])*math.prod(kernel)})

    def bicubic_aa(inputs, outputs):
        source, target = get_shape(inputs[0]), get_shape(outputs[0])
        h, w = source[-2:]
        oh, ow = target[-2:]
        # Conservative separable cubic filter estimate: support expands on
        # downsampling. Border clipping and coefficient construction excluded.
        tx = 2*math.ceil(2*max(1, w/ow))+1
        ty = 2*math.ceil(2*max(1, h/oh))+1
        return Counter({"bicubic_aa_estimate": math.prod(source[:-2]) *
                        (h*ow*(2*tx-1)+oh*ow*(2*ty-1))})

    class Forward(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, tensor):
            result = self.model(tensor)
            return result["pred_logits"], result["pred_boxes"]

    core.eval().to(device)
    dummy = torch.zeros(1, 3, size, size, device=device)
    analysis = FlopCountAnalysis(Forward(core), (dummy,))
    for name in ("fft_rfft2", "fft_irfft2", "fft_fft2", "fft_ifft2"):
        analysis.set_op_handle("aten::"+name, fft)
    costs = {"add": 1, "sub": 1, "mul": 1, "div": 1, "pow": 1,
             "sqrt": 1, "exp": 1, "log": 1, "log1p": 1, "sigmoid": 4,
             "gelu": 8, "silu": 5, "softmax": 5, "abs": 3,
             "mean": 1, "sum": 1, "neg": 1, "reciprocal": 1,
             "rsub": 1, "silu_": 5, "sin": 1, "cos": 1}
    for name, cost in costs.items():
        # Reduction cost depends on the input size, not the scalar output size.
        if name in {"mean", "sum"}:
            analysis.set_op_handle("aten::"+name,
                lambda inputs, outputs: Counter({"reduction": elements(inputs[0])}))
        else:
            analysis.set_op_handle("aten::"+name, pointwise(name, cost))
    analysis.set_op_handle("aten::scaled_dot_product_attention", attention)
    analysis.set_op_handle("aten::avg_pool2d", avg_pool)
    analysis.set_op_handle("aten::_upsample_bicubic2d_aa", bicubic_aa)
    # Data movement, masks/comparisons and tensor metadata have no floating math.
    ignored = ("fft_fftshift", "fft_ifftshift", "fft_fftfreq", "fft_rfftfreq",
               "real", "imag", "complex", "view_as_real", "view_as_complex",
               "lift_fresh", "resolve_conj", "resolve_neg", "where",
               "lt", "le", "gt", "ge", "eq", "ne", "any", "all",
               "logical_not", "bitwise_not", "meshgrid", "repeat_interleave",
               "repeat", "pad", "feature_dropout", "ones_like", "topk")
    # Eval feature_dropout is identity; pad is zero padding; topk is comparisons.
    analysis.set_op_handle("aten::linspace", pointwise("linspace", 2))
    for name in ignored:
        analysis.set_op_handle("aten::"+name, None)
    analysis.tracer_warnings("none").uncalled_modules_warnings(False)
    with torch.inference_mode():
        total = analysis.total()
    unsupported = dict(analysis.unsupported_ops())
    return {"GFLOPs": total/1e9 if not unsupported else None,
            "supported_gflops_subtotal": total/1e9,
            "unsupported_ops": unsupported,
            "complete_operator_coverage": not unsupported,
            "by_operator": dict(analysis.by_operator()),
            "parameters": sum(p.numel() for p in core.parameters()),
            "input_shape": [1, 3, size, size], "mode": "eval_fp32_forward_only",
            "convention": "fvcore: one fused multiply-add = one FLOP; FFT and nonlinear ops are analytic estimates",
            "fft_formula": "real 2.5*N*log2(HW), complex 5*N*log2(HW); normalization implementation cost excluded",
            "approximation_note": "Analytic estimate, not hardware instruction count. Nonlinear ops use declared fixed costs; antialiased bicubic uses conservative separable-filter support. Do not compare with a 2-FLOP-per-MAC table without reprofiling both models under one convention.",
            "scope": "normalized input through network incl DARM/DAFD/UFFR/MFA; excludes image resize, native detection postprocess and disk IO",
            "ignored_non_arithmetic_ops": list(ignored)}
