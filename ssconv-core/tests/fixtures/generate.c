// Synthetic fixtures matching ReShade's HDR color encodings.
// cc generate.c -ljxl -o generate && ./generate
#include <jxl/encode.h>
#include <assert.h>
#include <stdio.h>

static void encode(const char *name, int transfer, int alpha, int container) {
    JxlEncoder *enc = JxlEncoderCreate(NULL);
    JxlBasicInfo info;
    JxlEncoderInitBasicInfo(&info);
    info.xsize = 2;
    info.ysize = 1;
    info.bits_per_sample = 16;
    info.exponent_bits_per_sample = transfer == JXL_TRANSFER_FUNCTION_LINEAR ? 5 : 0;
    info.uses_original_profile = JXL_TRUE;
    if (alpha) {
        info.num_extra_channels = 1;
        info.alpha_bits = 16;
        info.alpha_exponent_bits = info.exponent_bits_per_sample;
    }
    assert(JxlEncoderUseContainer(enc, container) == JXL_ENC_SUCCESS);
    assert(JxlEncoderSetBasicInfo(enc, &info) == JXL_ENC_SUCCESS);
    JxlColorEncoding color;
    JxlColorEncodingSetToLinearSRGB(&color, JXL_FALSE);
    color.transfer_function = transfer;
    color.primaries = transfer == JXL_TRANSFER_FUNCTION_LINEAR ? JXL_PRIMARIES_SRGB : JXL_PRIMARIES_2100;
    assert(JxlEncoderSetColorEncoding(enc, &color) == JXL_ENC_SUCCESS);
    JxlEncoderFrameSettings *settings = JxlEncoderFrameSettingsCreate(enc, NULL);
    assert(JxlEncoderSetFrameLossless(settings, JXL_TRUE) == JXL_ENC_SUCCESS);
    // Binary16: [-0.25, 2, 12.5], [0, 0.5, 1]; alpha [0.5, 1].
    const uint16_t rgb[] = {0xb400, 0x4000, 0x4a40, 0, 0x3800, 0x3c00};
    const uint16_t rgba[] = {0xb400, 0x4000, 0x4a40, 0x3800, 0, 0x3800, 0x3c00, 0x3c00};
    const uint16_t hdr[] = {32768, 32768, 32768, 0, 65535, 16384};
    JxlPixelFormat format = {alpha ? 4 : 3,
        transfer == JXL_TRANSFER_FUNCTION_LINEAR ? JXL_TYPE_FLOAT16 : JXL_TYPE_UINT16,
        JXL_NATIVE_ENDIAN, 0};
    const void *pixels = transfer == JXL_TRANSFER_FUNCTION_LINEAR ? (alpha ? rgba : rgb) : hdr;
    assert(JxlEncoderAddImageFrame(settings, &format, pixels, alpha ? sizeof(rgba) : sizeof(rgb)) == JXL_ENC_SUCCESS);
    JxlEncoderCloseInput(enc);
    uint8_t bytes[4096], *next = bytes;
    size_t remaining = sizeof(bytes);
    assert(JxlEncoderProcessOutput(enc, &next, &remaining) == JXL_ENC_SUCCESS);
    FILE *file = fopen(name, "wb");
    assert(file);
    assert(fwrite(bytes, 1, next - bytes, file) == (size_t)(next - bytes));
    fclose(file);
    JxlEncoderDestroy(enc);
}

int main(void) {
    encode("scrgb.jxl", JXL_TRANSFER_FUNCTION_LINEAR, 0, 0);
    encode("scrgb-alpha.jxl", JXL_TRANSFER_FUNCTION_LINEAR, 1, 1);
    encode("pq.jxl", JXL_TRANSFER_FUNCTION_PQ, 0, 0);
    encode("hlg.jxl", JXL_TRANSFER_FUNCTION_HLG, 0, 1);
}
