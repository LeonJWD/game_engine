
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
};

const MAX_LIGHTS:u32=2;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) tangent_position: vec3<f32>,
    @location(2) tangent_view_position: vec3<f32>,
    @location(3)  world_normal:vec3<f32>,
    @location(4)  world_tangent:vec3<f32>,
    @location(5) world_bitangent:vec3<f32>,
    @location(6) world_position:vec4<f32>,
};

struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
};


@group(1) @binding(0)
var<uniform> camera: CameraUniform;

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@group(0)@binding(2)
var t_normal: texture_2d<f32>;
@group(0) @binding(3)
var s_normal: sampler;

@group(3) @binding(0)
var t_depth: texture_depth_2d;
@group(3) @binding(1)
var s_depth: sampler_comparison;

struct SpotLightUniform{
    view_pos: array<vec4<f32>,MAX_LIGHTS>,
    view_proj:array<mat4x4<f32>,MAX_LIGHTS>,
};


struct Light {
    position: array<vec4<f32>,MAX_LIGHTS>,
    color: array<vec4<f32>,MAX_LIGHTS>,
    num_lights: u32,
}
@group(2) @binding(0)
//var<uniform> light: Light;
var<uniform> spot_light_uniform: SpotLightUniform;


struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,

    @location(9) normal_matrix_0: vec3<f32>,
    @location(10) normal_matrix_1: vec3<f32>,
    @location(11) normal_matrix_2: vec3<f32>,
};


@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    let normal_matrix = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2,
    );

/*
    var light_position = array<vec3<f32>,MAX_LIGHTS>();
    var light_color = array<vec3<f32>,MAX_LIGHTS>();


    for (var i: u32 = 0; i < light.num_lights; i = i + 1) {
        light_position[i] = vec3<f32>(light.position[i][0], light.position[i][1], light.position[i][2]);
        light_color[i] = vec3<f32>(light.color[i][0], light.color[i][1], light.color[i][2]);
    }*/

    let world_normal = normalize(normal_matrix * model.normal);
    let world_tangent = normalize(normal_matrix * model.tangent);
    let world_bitangent = normalize(normal_matrix * model.bitangent);
    let tangent_matrix = transpose(mat3x3<f32>(
        world_tangent,
        world_bitangent,
        world_normal,
    ));

    

    let world_position = model_matrix * vec4<f32>(model.position, 1.0);

    var out: VertexOutput;
    out.world_normal=world_normal;
    out.world_bitangent=world_bitangent;
    out.world_tangent=world_tangent;
    out.clip_position = camera.view_proj * world_position;
    out.tex_coords = model.tex_coords;
    out.tangent_position = tangent_matrix * world_position.xyz;
    out.tangent_view_position = tangent_matrix * camera.view_pos.xyz;
    out.world_position=world_position;

    return out;
}

   @fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    let tangent_matrix = transpose(mat3x3<f32>(
        in.world_tangent,
        in.world_bitangent,
        in.world_normal,
    ));

    let object_color: vec4<f32>=textureSample(t_diffuse, s_diffuse, in.tex_coords);
    let object_normal: vec4<f32> = textureSample(t_normal, s_normal, in.tex_coords);

    let ambient_strength=0.5;
    let ambient_color=vec3<f32>(1.0,1.0,1.0)*ambient_strength;

    var light_position = array<vec3<f32>,MAX_LIGHTS>();
    var light_color = array<vec3<f32>,MAX_LIGHTS>();

    light_position[0] = vec3<f32>(spot_light_uniform.view_pos[0][0], spot_light_uniform.view_pos[0][1], spot_light_uniform.view_pos[0][2]);
    light_color[0] = vec3<f32>(1.0,1.0,1.0);

    var tangent_light_position = array<vec3<f32>,MAX_LIGHTS >();
    tangent_light_position[0] = tangent_matrix * light_position[0];

    let tangent_normal = object_normal.xyz * 2.0 - 1.0;

    var light_dir= array<vec3<f32>,MAX_LIGHTS>();

    light_dir[0] = normalize(tangent_light_position[0] - in.tangent_position);

    let view_dir = normalize(in.tangent_view_position - in.tangent_position);
    var half_light = array<vec3<f32>,MAX_LIGHTS>();
    half_light[0] = normalize(view_dir + light_dir[0]);

    var diffuse_strength = array<f32,MAX_LIGHTS>();
    diffuse_strength[0] = max(dot(tangent_normal, light_dir[0]), 0.0);

    var diffuse_color = vec3<f32>();
    diffuse_color = (light_color[0] * diffuse_strength[0]) + diffuse_color;

    var specular_color = vec3<f32>();
    let specular_strength = pow(max(dot(tangent_normal, half_light[0]), 0.0), 32.0);
    specular_color = specular_strength * light_color[0]+ specular_color;


    let d=sample_depth_map(in);
        



    return  vec4<f32>((ambient_color + diffuse_color + specular_color) * object_color.xyz *d,1.0);
} 

fn sample_depth_map(in: VertexOutput) ->f32{
    let bias=0.01;
    let m = spot_light_uniform.view_proj[0];
    let projcoords=m*in.world_position;
    let dm_coords=(projcoords.xyz/projcoords.w)*0.5 +0.5;
    var shadow=textureSampleCompare(t_depth,s_depth,dm_coords.xy,dm_coords.z);
    if(dm_coords.z >= 1.0){shadow = 0.0;} 
        
    return shadow;

}
