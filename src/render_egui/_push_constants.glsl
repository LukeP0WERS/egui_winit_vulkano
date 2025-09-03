#ifdef BINDLESS
#include <vulkano.glsl>
#endif

layout(push_constant) uniform PushConstants {
    #ifdef BINDLESS
    SampledImageId texture_id;
    SamplerId sampler_id;
    #endif
    vec2 screen_size;
    int output_in_linear_colorspace;
};