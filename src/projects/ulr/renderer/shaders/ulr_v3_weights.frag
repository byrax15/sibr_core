/*
 * Copyright (C) 2020, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */


#version 420

#define NUM_CAMS (12)
#define ULR_STREAMING (0)

in vec2 vertex_coord;
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_depth;
layout(location = 2) out vec4 out_normal;
layout(location = 3) out vec4 out_weights;

// 2D proxy texture.
layout(binding=0) uniform sampler2D proxy;

// Input cameras.
struct CameraInfos
{
  mat4 vp;
  mat4 v;
  vec3 pos;
  int selected;
  vec3 dir;
};
// They are stored in a contiguous buffer (UBO), lifting most limitations on the number of uniforms.
layout(std140, binding=4) uniform InputCameras
{
  CameraInfos cameras[NUM_CAMS];
};

// Uniforms.
uniform int camsCount;
uniform vec3 ncam_pos;
uniform bool occ_test = true;
uniform bool invert_mask = false;
uniform bool is_binary_mask = true;
uniform bool discard_black_pixels = true;
uniform bool doMasking = false;
uniform bool flipRGBs = false;
uniform bool showWeights = false;
uniform float epsilonOcclusion = 1e-5;
uniform bool winner_takes_all = false;
uniform int selectedCam;

// for uv derivatives blending
uniform bool useUVDerivatives = false;
uniform float uvDerivativesAlphaBlending = 0.5f;
uniform float uvDerivativesScaleFactor = 1.0f;
uniform vec2 rtResolution = vec2(1.0);

#define INFTY_W 100000.0
#define BETA 	1e-1  	/* Relative importance of resolution penalty */

// Textures.
// To support both the regular version (using texture arrays) and the streaming version (using 2D RTs),
// we wrap the texture accesses in two helpers that hide the difference.

#if ULR_STREAMING

uniform sampler2D input_rgbds[NUM_CAMS];
uniform sampler2D input_masks[NUM_CAMS];

vec4 getRGBD(vec3 xy_camid){
	if(flipRGBs){
		xy_camid.y = 1.0 - xy_camid.y;
	}
	vec4 rgbd = texture(input_rgbds[int(xy_camid.z)], xy_camid.xy);
	if(flipRGBs){
		xy_camid.y = 1.0 - xy_camid.y;
	}
	return rgbd;
}

float getMask(vec3 xy_camid){
	return texture(input_masks[int(xy_camid.z)], xy_camid.xy).r;
}

#else

layout(binding=1) uniform sampler2DArray input_rgbs;
layout(binding=2) uniform sampler2DArray input_depths;
layout(binding=3) uniform sampler2DArray input_masks;

// 2D proxy normals.
layout(binding=5) uniform sampler2D normals;
layout(binding=6) uniform sampler2D normalsCam;

struct CostIndex
{
	float cost;
	vec3 color;
	float dist;
};

CostIndex bin_list[NUM_CAMS];

void bubbleSort(int fullSize, int sortSize, float maxDist)
{
	CostIndex temp;
	for (int i = 0; i < sortSize; i++)
	{
		bool sorted = true;
		for (int j = fullSize - 1; j > 0; j--)
		{
			if (bin_list[j].cost + (bin_list[j].dist/maxDist) < bin_list[j-1].cost + (bin_list[j-1].dist/maxDist))
			{
				temp = bin_list[j];
				bin_list[j] = bin_list[j-1];
				bin_list[j-1] = temp;
				sorted = false;
			}
		}
		if (sorted) return;
	}
}


vec4 getRGBD(vec3 xy_camid){
	if(flipRGBs){
		xy_camid.y = 1.0 - xy_camid.y;
	}
	vec3 rgb = texture(input_rgbs, xy_camid).rgb;
	if(flipRGBs){
		xy_camid.y = 1.0 - xy_camid.y;
	}
	float depth = texture(input_depths, xy_camid).r;
    return vec4(rgb,depth);
}

float getMask(vec3 xy_camid){
	return texture(input_masks, xy_camid).r;
}

#endif

// Helpers.
vec3 project_unnormalized(vec3 point, mat4 proj) {
  vec4 p1 = proj * vec4(point, 1.0);
  return p1.xyz;
}

vec3 project(vec3 point, mat4 proj) {
  vec4 p1 = proj * vec4(point, 1.0);
  vec3 p2 = (p1.xyz/p1.w);
  return (p2.xyz*0.5 + 0.5);
}

bool frustumTest(vec3 p, vec2 ndc, int i) {
  vec3 d1 = cameras[i].dir;
  vec3 d2 = p - cameras[i].pos;
  return !any(greaterThan(ndc, vec2(1.0))) && dot(d1,d2)>0.0;
}

vec3 getRandomColor(int x);

void main(void){
  		
  vec4 point = texture(proxy, vertex_coord);
  vec4 normalW = texture(normals, vertex_coord);	// [-1, 1]
  vec4 normalC = texture(normalsCam, vertex_coord);	// [-1, 1]
  float depth = point.w;
  if ( point.w >= 1.0) {
	discard;
  }
  vec4  color0 = vec4(0.0,0.0,0.0,INFTY_W);
  vec4  color1 = vec4(0.0,0.0,0.0,INFTY_W);
  vec4  color2 = vec4(0.0,0.0,0.0,INFTY_W);
  vec4  color3 = vec4(0.0,0.0,0.0,INFTY_W);
  bool atLeastOneValid = false;
  
  vec3 norms, view_dir;
  int counter = 0;
  float maxDist = -1.0;
  for(int i = 0; i < NUM_CAMS; i++){
	
	
	if(cameras[i].selected == 0){
		continue;
	}

	vec3 uvd = project(point.xyz, cameras[i].vp);
	vec2 ndc = abs(2.0*uvd.xy-1.0);

	vec2 uv_ddx = dFdx(uvd.xy * rtResolution); 
	vec2 uv_ddy = dFdy(uvd.xy * rtResolution);


	if (frustumTest(point.xyz, ndc, i)){
		vec3 xy_camid = vec3(uvd.xy,i);
		vec4 color = getRGBD(xy_camid);

		if(doMasking){        
			float masked = getMask(xy_camid);
             
            if( invert_mask ){
                masked = 1.0 - masked;
            }
            
            if( is_binary_mask && camsCount != i ){
                if( masked <= 0.05) {
                    continue;
                }
			} 
			
		}


		if (occ_test){
			if(abs(uvd.z-color.w) >= epsilonOcclusion) {	  
				continue;
			}
		}
		float penaltyValue = 0;

		// compute ang deviation, ulr weights
		vec3 v1 = (point.xyz - cameras[i].pos);
		vec3 v2 = (point.xyz - ncam_pos);
		float dist_i2p 	= length(v1);
		float dist_n2p 	= length(v2);

		float penalty_ang = float(occ_test) * max(0.0001, acos(dot(v1,v2)/(dist_i2p*dist_n2p)));
		float penalty_res = max(0.0001, (dist_i2p - dist_n2p)/dist_i2p );
		 
		view_dir = normalize(-v1); //project(-v1, cameras[i].vp);
		norms =  normalize(normalW.xyz); //project(normalW.xyz, cameras[i].vp);
		//penaltyValue = penalty_ang + BETA*penalty_res;
		if (maxDist < dist_i2p){
			maxDist = dist_i2p;
		}

		penaltyValue = (1.0 - dot(view_dir, norms));

		if(showWeights){
			
			//normalC =  project(normalW.xyz, cameras[camsCount].vp);
			color.xyz = vec3(dot(view_dir, norms));
		}
		atLeastOneValid = true;
		bin_list[counter] = CostIndex(penaltyValue, color.xyz, dist_i2p);
		counter++;
        //color.w = penaltyValue;  
		//color0=color;
		// compare with best four candiates and insert at the
		// appropriate rank
//		if (color.w<color3.w) {    // better than fourth best candidate
//			if (color.w<color2.w) {    // better than third best candidate
//				color3 = color2;
//				if (color.w<color1.w) {    // better than second best candidate
//					color2 = color1;
//					if (color.w<color0.w) {    // better than best candidate
//						color1 = color0;
//						color0 = color;
//					} else {
//						color1 = color;
//					}
//				} else {
//					color2 = color;
//				}
//			} else {
//				color3 = color;
//			}
//		}
	  }  
    }
	bubbleSort(counter, 20, maxDist);
   
    if(!atLeastOneValid){
         discard;
    }
//	float thresh = 1.0000001 * color3.w;
//    color0.w = max(0, 1.0 - color0.w/thresh);
//    color1.w = max(0, 1.0 - color1.w/thresh);
//    color2.w = max(0, 1.0 - color2.w/thresh);
//    color3.w = 1.0 - 1.0/1.0000001;
//
//    // ignore any candidate which is uninit
//	if (color0.w == INFTY_W) color0.w = 0;
//    if (color1.w == INFTY_W) color1.w = 0;
//    if (color2.w == INFTY_W) color2.w = 0;
   
    // output weights
	out_color = (selectedCam < counter) ? vec4(bin_list[selectedCam].color, 1.0) : vec4(0.0);
//	if(selectedCam == 0){
//		out_color.xyz = color0.xyz;
//	}
//	else if(selectedCam == 1){
//		out_color.xyz = color1.xyz;
//	}
//	else if(selectedCam == 2){
//		out_color.xyz = color2.xyz;
//	}
//	else if(selectedCam == 3){
//		out_color.xyz = color3.xyz;
//	}
//	else{
//		out_color.xyz = vec3(0.0);
//	}
	//out_color.w = 1.0;
	out_depth.xyz = vec3(pow(depth, 1));
	out_depth.w = 1.0;
	out_normal.xyz = normalC.xyz;
	out_normal.w = 1.0;
	out_weights.xyz = vec3(color0.w);
	out_weights.w = 1.0;
    gl_FragDepth = point.w;
	
}






/*
float getPenalizeStretch(vec2 uv)
{
	//uv = uv * reso;
      // Source:
      // - Hyperlapse papers [Kopf et al. 2014]
      // - http://www.lucidarme.me/?p=4624

      mat2 jacobian = mat2(
        dFdx(uv),
        dFdy(uv)
        );

      float a = jacobian[0][0];
      float b = jacobian[1][0];
      float c = jacobian[0][1];
      float d = jacobian[1][1];
      float aa = a*a;
      float bb = b*b;
      float cc = c*c;
      float dd = d*d;

      float S1 = aa + bb + cc + dd;
      float S1a = (aa+bb-cc-dd);
      float S1b = (a*c + b*d);
      float S2 = sqrt(S1a*S1a + 4*S1b*S1b);

      vec2  sigma = vec2(sqrt((S1+S2)/2.0), sqrt((S1-S2)/2.0));
      return 1.0 - min(sigma.x, sigma.y)/max(sigma.x, sigma.y);
}
*/


// Random number generation:
// "Quality hashes collection" (https://www.shadertoy.com/view/Xt3cDn)
// by nimitz 2018 (twitter: @stormoid)
// The MIT License
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/** Compute the based hash for a given index.
	\param p the index
	\return the hash
*/
uint baseHash(uint p) {
	p = 1103515245U*((p >> 1U)^(p));
	uint h32 = 1103515245U*((p)^(p>>3U));
	return h32^(h32 >> 16);
}

/** Generate a random vec3 from an index seed (see http://random.mat.sbg.ac.at/results/karl/server/node4.html).
	\param x the seed
	\return a random vec3
*/
vec3 getRandomColor(int x) {
	// Color 0 is black, so we shift everything.
	x = x+1;
	uint n = baseHash(uint(x));
	uvec3 rz = uvec3(n, n*16807U, n*48271U);
	return vec3(rz & uvec3(0x7fffffffU))/float(0x7fffffff);
}