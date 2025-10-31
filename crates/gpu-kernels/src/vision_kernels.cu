extern "C" __global__ void bgr_resize_normalize(
    const unsigned char* input,
    int in_width,
    int in_height,
    int out_width,
    int out_height,
    float* out_tensor,
    unsigned char* out_bgr
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = (unsigned int)(out_width * out_height);
    if (idx >= total) {
        return;
    }

    const int dst_x = idx % out_width;
    const int dst_y = idx / out_width;

    const float scale_x = (float)in_width / (float)out_width;
    const float scale_y = (float)in_height / (float)out_height;

    float src_x = ((float)dst_x + 0.5f) * scale_x - 0.5f;
    float src_y = ((float)dst_y + 0.5f) * scale_y - 0.5f;

    int x0 = (int)floorf(src_x);
    int y0 = (int)floorf(src_y);
    float tx = src_x - (float)x0;
    float ty = src_y - (float)y0;

    if (x0 < 0) {
        x0 = 0;
        tx = 0.0f;
    }
    if (y0 < 0) {
        y0 = 0;
        ty = 0.0f;
    }
    int x1 = min(x0 + 1, in_width - 1);
    int y1 = min(y0 + 1, in_height - 1);

    const int stride = in_width * 3;
    const unsigned char* row0 = input + y0 * stride;
    const unsigned char* row1 = input + y1 * stride;

    const int idx00 = x0 * 3;
    const int idx10 = x1 * 3;

    const unsigned char* p00 = row0 + idx00;
    const unsigned char* p10 = row0 + idx10;
    const unsigned char* p01 = row1 + idx00;
    const unsigned char* p11 = row1 + idx10;

    const float w00 = (1.0f - tx) * (1.0f - ty);
    const float w10 = tx * (1.0f - ty);
    const float w01 = (1.0f - tx) * ty;
    const float w11 = tx * ty;

    float b = w00 * (float)p00[0] + w10 * (float)p10[0] + w01 * (float)p01[0] +
              w11 * (float)p11[0];
    float g = w00 * (float)p00[1] + w10 * (float)p10[1] + w01 * (float)p01[1] +
              w11 * (float)p11[1];
    float r = w00 * (float)p00[2] + w10 * (float)p10[2] + w01 * (float)p01[2] +
              w11 * (float)p11[2];

    const float inv255 = 1.0f / 255.0f;
    r *= inv255;
    g *= inv255;
    b *= inv255;

    const unsigned int plane_stride = (unsigned int)(out_width * out_height);
    out_tensor[idx] = r;
    out_tensor[idx + plane_stride] = g;
    out_tensor[idx + plane_stride * 2] = b;

    unsigned char* dst = out_bgr + idx * 3;
    dst[0] = (unsigned char)fminf(fmaxf(b * 255.0f, 0.0f), 255.0f);
    dst[1] = (unsigned char)fminf(fmaxf(g * 255.0f, 0.0f), 255.0f);
    dst[2] = (unsigned char)fminf(fmaxf(r * 255.0f, 0.0f), 255.0f);
}

__device__ __forceinline__ float iou_single(
    float x1,
    float y1,
    float x2,
    float y2,
    float x1b,
    float y1b,
    float x2b,
    float y2b
) {
    const float ix1 = fmaxf(x1, x1b);
    const float iy1 = fmaxf(y1, y1b);
    const float ix2 = fminf(x2, x2b);
    const float iy2 = fminf(y2, y2b);
    const float iw = fmaxf(0.0f, ix2 - ix1);
    const float ih = fmaxf(0.0f, iy2 - iy1);
    const float intersection = iw * ih;
    if (intersection <= 0.0f) {
        return 0.0f;
    }
    const float area_a = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    const float area_b = fmaxf(0.0f, x2b - x1b) * fmaxf(0.0f, y2b - y1b);
    const float denom = area_a + area_b - intersection + 1e-6f;
    return intersection / denom;
}

extern "C" __global__ void nms_suppress(
    unsigned long long boxes_ptr,
    int num_boxes,
    float iou_threshold,
    int* keep_flags
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const float* boxes = reinterpret_cast<const float*>(boxes_ptr);
    for (int i = 0; i < num_boxes; ++i) {
        keep_flags[i] = 1;
    }
    for (int i = 0; i < num_boxes; ++i) {
        if (keep_flags[i] == 0) {
            continue;
        }
        const float x1 = boxes[i * 4 + 0];
        const float y1 = boxes[i * 4 + 1];
        const float x2 = boxes[i * 4 + 2];
        const float y2 = boxes[i * 4 + 3];
        for (int j = i + 1; j < num_boxes; ++j) {
            if (keep_flags[j] == 0) {
                continue;
            }
            const float x1b = boxes[j * 4 + 0];
            const float y1b = boxes[j * 4 + 1];
            const float x2b = boxes[j * 4 + 2];
            const float y2b = boxes[j * 4 + 3];
            const float overlap = iou_single(x1, y1, x2, y2, x1b, y1b, x2b, y2b);
            if (overlap > iou_threshold) {
                keep_flags[j] = 0;
            }
        }
    }
}

__device__ __forceinline__ unsigned char glyph_row(char ch, int row) {
    switch (ch) {
        case 'A': {
            const unsigned char rows[7] = {0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001};
            return rows[row];
        }
        case 'C': {
            const unsigned char rows[7] = {0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110};
            return rows[row];
        }
        case 'E': {
            const unsigned char rows[7] = {0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b11111};
            return rows[row];
        }
        case 'F': {
            const unsigned char rows[7] = {0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b10000};
            return rows[row];
        }
        case 'M': {
            const unsigned char rows[7] = {0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001};
            return rows[row];
        }
        case 'N': {
            const unsigned char rows[7] = {0b10001, 0b11001, 0b10101, 0b10101, 0b10011, 0b10001, 0b10001};
            return rows[row];
        }
        case 'O': {
            const unsigned char rows[7] = {0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110};
            return rows[row];
        }
        case 'P': {
            const unsigned char rows[7] = {0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000};
            return rows[row];
        }
        case 'R': {
            const unsigned char rows[7] = {0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001};
            return rows[row];
        }
        case 'S': {
            const unsigned char rows[7] = {0b01111, 0b10000, 0b01110, 0b00001, 0b00001, 0b10001, 0b01110};
            return rows[row];
        }
        case '0': {
            const unsigned char rows[7] = {0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110};
            return rows[row];
        }
        case '1': {
            const unsigned char rows[7] = {0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110};
            return rows[row];
        }
        case '2': {
            const unsigned char rows[7] = {0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111};
            return rows[row];
        }
        case '3': {
            const unsigned char rows[7] = {0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110};
            return rows[row];
        }
        case '4': {
            const unsigned char rows[7] = {0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010};
            return rows[row];
        }
        case '5': {
            const unsigned char rows[7] = {0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110};
            return rows[row];
        }
        case '6': {
            const unsigned char rows[7] = {0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110};
            return rows[row];
        }
        case '7': {
            const unsigned char rows[7] = {0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000};
            return rows[row];
        }
        case '8': {
            const unsigned char rows[7] = {0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110};
            return rows[row];
        }
        case '9': {
            const unsigned char rows[7] = {0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100};
            return rows[row];
        }
        case '%': {
            const unsigned char rows[7] = {0b10001, 0b10010, 0b00100, 0b01000, 0b10010, 0b10001, 0b00000};
            return rows[row];
        }
        case '.': {
            const unsigned char rows[7] = {0, 0, 0, 0, 0, 0b00110, 0b00110};
            return rows[row];
        }
        case ' ': {
            return 0;
        }
        default:
            return 0;
    }
}

__device__ __forceinline__ void blend_pixel(
    unsigned char* image,
    int width,
    int height,
    int x,
    int y,
    unsigned char r,
    unsigned char g,
    unsigned char b,
    unsigned char a
) {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return;
    }
    unsigned char* dst = image + ((size_t)y * width + x) * 3u;
    const float alpha = (float)a / 255.0f;
    const float inv_alpha = 1.0f - alpha;
    float db = (float)dst[0];
    float dg = (float)dst[1];
    float dr = (float)dst[2];
    dst[0] = (unsigned char)(alpha * (float)b + inv_alpha * db);
    dst[1] = (unsigned char)(alpha * (float)g + inv_alpha * dg);
    dst[2] = (unsigned char)(alpha * (float)r + inv_alpha * dr);
}

__device__ void draw_text_line(
    unsigned char* image,
    int width,
    int height,
    int origin_x,
    int origin_y,
    const unsigned char* text,
    int length,
    unsigned char r,
    unsigned char g,
    unsigned char b
) {
    int cursor = origin_x;
    for (int idx = 0; idx < length; ++idx) {
        const char ch = (char)text[idx];
        for (int row = 0; row < 7; ++row) {
            const unsigned char pattern = glyph_row(ch, row);
            const int py = origin_y + row;
            if (py < 0 || py >= height) {
                continue;
            }
            for (int col = 0; col < 5; ++col) {
                if ((pattern >> (4 - col)) & 1) {
                    const int px = cursor + col;
                    blend_pixel(image, width, height, px, py, r, g, b, 255u);
                }
            }
        }
        cursor += 6;
    }
}

extern "C" __global__ void annotate_overlays(
    unsigned char* image,
    int width,
    int height,
    const int* boxes,
    const int* label_positions,
    const int* label_offsets,
    const int* label_lengths,
    const unsigned char* label_chars,
    int num_boxes,
    const unsigned char* info_text,
    int info_len,
    int info_x,
    int info_y
) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_boxes && gid > 0) {
        return;
    }

    const unsigned char edge_r = 0u;
    const unsigned char edge_g = 255u;
    const unsigned char edge_b = 0u;

    if (gid < num_boxes) {
        const int left = max(0, boxes[gid * 4 + 0]);
        const int top = max(0, boxes[gid * 4 + 1]);
        const int right = min(width - 1, boxes[gid * 4 + 2]);
        const int bottom = min(height - 1, boxes[gid * 4 + 3]);

        for (int x = left; x <= right; ++x) {
            blend_pixel(image, width, height, x, top, edge_r, edge_g, edge_b, 255u);
            blend_pixel(image, width, height, x, bottom, edge_r, edge_g, edge_b, 255u);
        }
        for (int y = top; y <= bottom; ++y) {
            blend_pixel(image, width, height, left, y, edge_r, edge_g, edge_b, 255u);
            blend_pixel(image, width, height, right, y, edge_r, edge_g, edge_b, 255u);
        }

        const int label_x = label_positions[gid * 2 + 0];
        const int label_y = label_positions[gid * 2 + 1];
        const int label_len = label_lengths[gid];
        const int label_offset = label_offsets[gid];
        const unsigned char* text = label_chars + label_offset;
        const int background_width = label_len * 6;

        for (int py = 0; py < 8; ++py) {
            for (int px = 0; px < background_width; ++px) {
                blend_pixel(image, width, height, label_x + px, label_y + py, 0u, 0u, 0u, 180u);
            }
        }

        draw_text_line(image, width, height, label_x, label_y, text, label_len, edge_r, edge_g, edge_b);
    }

    if (gid == 0 && info_len > 0) {
        const int info_width = info_len * 6;
        for (int py = 0; py < 8; ++py) {
            for (int px = 0; px < info_width + 4; ++px) {
                blend_pixel(image, width, height, info_x + px, info_y + py, 0u, 0u, 0u, 180u);
            }
        }
        draw_text_line(image, width, height, info_x + 2, info_y, info_text, info_len, 255u, 255u, 255u);
    }
}
