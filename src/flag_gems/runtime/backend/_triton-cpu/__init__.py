from backend_utils import VendorInfoBase  # noqa: E402

TRITON_CPU_VAILD_CMD = ("bash", "-c", "[[ \"$TRITON_CPU_BACKEND\" -eq 1 ]] || { exit 1; }")

vendor_info = VendorInfoBase(
    vendor_name="triton-cpu", device_name="cpu", device_query_cmd=TRITON_CPU_VAILD_CMD
)

CUSTOMIZED_UNUSED_OPS = ("cumsum", "cos", "add")


__all__ = ["*"]
