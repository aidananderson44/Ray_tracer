#version 430 core
out vec4 color;
in vec2 uv;
uniform vec3 light_pos[20];
uniform int num_lights;
uniform int num_objects;
uniform int num_triangles;
uniform vec3 U;
uniform vec3 V;
uniform vec3 W;
uniform vec3 S;
uniform vec3 E;
uniform float left;
uniform float right;
uniform float bottom;
uniform float top;
#define WHITE vec3(1, 1, 1)
#define BLACK vec3(0,0,0)
#define epsilon 0.0005
#define PI 3.14159265359f
struct obj{
	float type;

	float posx;
	float posy;
	float posz;
	float size1;
	float size2;
//	float size3;
	float c1;
	float c2;
	float c3;
	float dirx;
	float diry;
	float dirz;
	//vec3 test;
};
struct tr
{
	
	float pos1x;
	float pos1y;
	float pos1z;
	float pos2x;
	float pos2y;
	float pos2z;
	float pos3x;
	float pos3y;
	float pos3z;

	float c1;
	float c2;
	float c3;
	float normx;
	float normy;
	float normz;



};
struct stack_obj
{
	vec3 ray;
	vec3 E;
	float intensity;
};
const int num_iter = 5;
stack_obj stack[num_iter];
int stack_back = 0;
int stack_front = 0;
layout (std140) uniform object_data
{
	obj object[99];
};
layout (std140) uniform triangle_data
{
	tr triangle[99];
};

bool is_inside(vec3 pos, vec3 ray, obj o)
{
	if(int(o.type) == 1)
		return false;
	else
	{
		return false;
		
		vec3 sphere_pos = vec3(o.posx, o.posy, o.posz);
		float radius = o.size1;
		float dist = length(pos - sphere_pos);
		vec3 normal = (pos - sphere_pos) / radius;
		if (dist < radius + 0.5 && dot(ray, normal) > 0)
			return true;
		else
			return false;
			
	}
}
vec3 sphere_normal(vec3 pos, vec3 sphere_pos, float radius, vec3 E, bool inside)
{
	vec3 normal = (pos - sphere_pos) / radius;
	if(inside)
		return -normal;
	else
		return normal;
}

vec3 get_normal(vec3 pos, int i, vec3 E, vec3 ray)
{
	if(i < num_objects)
	{
		obj o = object[i];
		if(int(o.type) == 0)
			return sphere_normal(pos, vec3(o.posx, o.posy, o.posz), o.size1, E,is_inside(pos, ray, o));
		else
			return vec3(o.dirx, o.diry, o.dirz);
	}
	else
	{
		int d = i - num_objects;
		return	vec3(triangle[d].normx, triangle[d].normy, triangle[d].normz );
	}
}
float sphere_intersect(vec3 ray, vec3 E,vec3 sphere_pos, float radius)
{
	vec3 EsubC = E - sphere_pos;
	float disc = pow(dot(ray, EsubC), 2) - dot(EsubC, EsubC) + radius * radius;
	if(disc > 0)
	{
		float t_m = (-(dot(ray, EsubC) + sqrt(disc)));
	//	if(dot(ray, ray*t_m) > 0)
			return t_m;
	//	else
		//	return (-(dot(ray, EsubC) - sqrt(disc)));
	

	}
	return -1;
}
float plane_intersect(vec3 ray, vec3 E,vec3 plane_pos, vec3 plane_normal)
{
	
	float denom = dot(ray, plane_normal);
	if (denom == 0)
		return -1;
	else {		
		float t = dot((plane_pos - E), plane_normal) / denom;
		return t;		
	}
}
//https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
//https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
float triangle_intersect(vec3 ray, vec3 E, vec3 vertex0, vec3 vertex1, vec3 vertex2, vec3 norm)
{
	float t = plane_intersect(ray, E, vertex0, norm);
	if(t < 0) return t;

	vec3 p = E + t*ray;

	vec3 edge0 = vertex1 - vertex0;
	vec3 edge1 = vertex2 - vertex1;
	vec3 edge2 = vertex0 - vertex2;

	vec3 c0 = p - vertex0;
	vec3 c1 = p - vertex1;
	vec3 c2 = p - vertex2;

	if(dot(norm, cross(edge0, c0)) > 0 &&
	   dot(norm, cross(edge1, c1)) > 0 &&
	   dot(norm, cross(edge2, c2)) > 0)
	   return t;
	else return -1;

}
float intersect(vec3 ray, vec3 E, int i)
{
	if(i < num_objects)
	{
		obj o = object[i];
		if(int(o.type) == 0)
			return sphere_intersect(ray, E, vec3(o.posx, o.posy, o.posz), o.size1);
		else
			return plane_intersect(ray, E,vec3(o.posx, o.posy, o.posz), vec3(o.dirx, o.diry, o.dirz));
	}
	else
	{
		int d = i - num_objects;
		vec3 pos1 = vec3(triangle[d].pos1x, triangle[d].pos1y, triangle[d].pos1z );
		vec3 pos2 = vec3(triangle[d].pos2x, triangle[d].pos2y, triangle[d].pos2z );
		vec3 pos3 = vec3(triangle[d].pos3x, triangle[d].pos3y, triangle[d].pos3z );
		vec3 norm = vec3(triangle[d].normx, triangle[d].normy, triangle[d].normz );
		return triangle_intersect(ray, E, pos1, pos2, pos3, norm);
	}
}


vec3 plane_color(vec3 pos, vec3 plane_pos)
{
	int checker_board_size = 1;
	vec3 obj_pos = pos - plane_pos;
	float width = dot(obj_pos, vec3(1, 0, 0));
	float height = dot(obj_pos, vec3(0, 0, 1));

	int on = 0;
	int off = 1;
	int w_on = 0;
	int w_off = 1;

	if (width < 0)
	{
		w_on = 1;
		w_off = 0;
	}
	if (height < 0)
	{
		on = 1;
		off = 0;
	}
	if (abs(int(checker_board_size*width)) % 2 == w_on)
	{
		if (abs(int(checker_board_size*height)) % 2 == on)
			return BLACK;
		else if (abs(int(checker_board_size*height)) % 2 == off)
			return WHITE;
	}		
	else if (abs(int(checker_board_size*width)) % 2 == w_off)
	{
		if (abs(int(checker_board_size*height)) % 2 == off)
			return BLACK;
		else if (abs(int(checker_board_size*height)) % 2 == on)
			return WHITE;
	}

}

vec3 get_colour(vec3 pos, int i)
{
	if(i < num_objects)
	{
		obj o = object[i];
		if(int(o.type) == 1)
			return plane_color(pos, vec3(o.posx, o.posy, o.posz));
		else
			return vec3(o.c1, o.c2, o.c3);
	}
	else
		return vec3(triangle[i - num_objects].c1, triangle[i - num_objects].c2,triangle[i - num_objects].c3);
}
int find_closest(vec3 ray, vec3 E)
{

	int i;
	float closest = 1.0/0.0; 
	int closest_index = -1;
	for(i = 0; i < num_objects; i++)
	{
		float t = intersect(ray, E, i);
		if(t >= 0 && t < closest)
		{
			closest = t;
			closest_index = i;

		}
	}
	for(i = 0; i < num_triangles; i++)
	{
		vec3 pos1 = vec3(triangle[i].pos1x, triangle[i].pos1y, triangle[i].pos1z );
		vec3 pos2 = vec3(triangle[i].pos2x, triangle[i].pos2y, triangle[i].pos2z );
		vec3 pos3 = vec3(triangle[i].pos3x, triangle[i].pos3y, triangle[i].pos3z );
		vec3 norm = vec3(triangle[i].normx, triangle[i].normy, triangle[i].normz );
		float t = triangle_intersect(ray, E, pos1, pos2, pos3, norm);
		if(t >= 0 && t < closest)
		{
			closest = t;
			closest_index = i + num_objects;
		}

	}
	return closest_index;

}
bool in_shadow(vec3 pos, vec3 l_p)
{
	vec3 lightDir = normalize(l_p - pos);
	int i = find_closest(lightDir, pos + lightDir*epsilon);
	if(i == -1)
		return false;
	else
	{
		float light_dist = length(l_p - pos);
		float t = intersect(lightDir, pos, i);
		if(t < light_dist)
			return true;
		else
			return false;

	}
}
vec3 lighting(vec3 normal, vec3 ray, vec3 colour,vec3 pos, vec3 l_p)
{
	if(in_shadow(pos, l_p))
		return vec3(colour* 0.1);
	vec3 lightDir = normalize(l_p - pos);
	vec3 diffuse = max(0, dot(normal, lightDir))*colour;
	vec3 H = normalize((-ray + lightDir));
	float Is = max(pow(dot(normal, H), 16)*1, 0);
	return vec3(diffuse + Is*vec3(1, 1, 1) + 0.25*colour);
}
void push(vec3 ray, vec3 E, float intensity)
{
	if(stack_back < num_iter)
	{
		stack[stack_back].ray = ray;
		stack[stack_back].E = E;
		stack[stack_back].intensity = intensity;
		stack_back+=1;
	}
}
stack_obj pop()
{
	stack_front +=1;
	return stack[stack_front -1];
}
float get_refractive(obj o)
{
	if(int(o.type) == 0)
		return 1;
	else
		return 0;
}

vec3 trace(vec3 init_ray, vec3 init_E)
{

	vec3 final = vec3(0,0,0);
	vec3 ray = init_ray;
	vec3 E = init_E;
	push(ray, E, 1);
	while(stack_front <= stack_back)
	{
		stack_obj obj = pop();
		ray = obj.ray;
		E = obj.E;
		float intensity = obj.intensity;
		int i = find_closest(ray, E);
		if(i == -1)
			break;	

		vec3 pos = E + ray*intersect(ray, E, i);
		vec3 normal = get_normal(pos, i, E, ray);
		vec3 colour = get_colour(pos, i);
		vec3 l = vec3(0,0,0);
		int k;
		for(k = 0; k < num_lights; k++)
			l +=lighting(normal,ray, colour, pos, light_pos[k]) * intensity;
		
		l /= num_lights;
		final += intensity*l * pow(0.75, stack_back);

		
		float n1 = 1.0;
		float n2 = 1.1;
		float R0 = ((n1 - n2)/(n1 + n2)) * ((n1 - n2)/(n1 + n2));
		float R_t = R0 + (1 - R0)* pow((1 - dot(normal, -ray)), 5);
		/*
		if(get_refractive(object[i]) == 1)
		{
			float r =  n1/n2;
			if(!is_inside(pos, ray, object[i]))
				r = n2/n1;
				
			float c = dot(-normal, ray);
			vec3 refraction_ray = r * ray + (r*c - sqrt(1 - r*r*(1 - (c*c))))* normal;
			vec3 refraction_E = pos + refraction_ray*epsilon;
			push(refraction_ray, refraction_E, 0.5);
		}*/
	//	else{
		
			vec3 reflection_ray = ray - 2 * normal*(dot(normal, ray));
			vec3 reflection_E = pos + reflection_ray*epsilon;
			push(reflection_ray, reflection_E, 1);//0.5);
		//}

	
	}
	stack_back = 0;
	stack_front = 0;
	return final/1.5;
}

float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}
vec2 rand_polar(vec2 seed, float radius)
{
	float theta = 2*PI *rand(seed);
	seed*=theta;
	float r = radius*sqrt(rand(seed));
	return vec2(r, theta);
	
}

void main() 
{

	

	
	vec3 pixel = left * U + (right - left)*uv.x* U;
	pixel += bottom * V + (top - bottom)*uv.y*V;
	
	vec3 ray = normalize(pixel + S - E);
	color = vec4(1.5*trace(ray, E), 1);
//	color = vec4(triangle[1].c1,triangle[1].c2,triangle[1].c3 , 1);
	
/*	int i = 1;
//	color = vec4(triangle[i].pos3x, triangle[i].pos3y , triangle[i].pos3z, 1);
	vec3 pos1 = vec3(triangle[i].pos1x, triangle[i].pos1y, triangle[i].pos1z );
	vec3 pos2 = vec3(triangle[i].pos2x, triangle[i].pos2y, triangle[i].pos2z );
	vec3 pos3 = vec3(triangle[i].pos3x, triangle[i].pos3y, triangle[i].pos3z );
	vec3 norm = vec3(triangle[i].normx, triangle[i].normy, triangle[i].normz );
	float t = triangle_intersect(ray, E, pos1, pos2, pos3, norm);
	if(t > 0)
		color = vec4(1, 0, 0, 1);
	else if(t == -1)
		color = vec4(0, 1, 0, 1);
	else if(t == -2)
		color = vec4(0, 0, 1, 1);
	else if(t == -3)
		color = vec4(1, 1, 1, 1);*/

	//


	/*
	vec3 middle = left * U + (right - left)*0.5* U;
	middle += pixel += bottom * V + (top - bottom)*0.5*V;
	vec3 middle_ray = normalize(middle + S - E);
	int d = find_closest(middle_ray, E);
	float focal_distance = 5;
	if(d >= 0)
		focal_distance = intersect(middle_ray, E, object[d]);
	vec3 ray = normalize(pixel + S - E);
	//color = vec4(trace(ray, E), 1);
	*/
	
/*	float aperture = 0.2;
	
//	vec3 focal_plane = focal_distance*W + E;
	vec3 focal_plane = 10*W + E;
	float t = plane_intersect(ray, E, focal_plane, -W);
	vec3 focal_point = E + t*ray;
	int i;
	vec2 seed = uv;
	vec3 blur_color = vec3(0,0,0);


	int num_blur = 3;

	for(i = 0; i < num_blur; i++)
	{
		
	//	vec2 coords = rand_polar(seed, aperture);
	//	float x = coords.x*cos(coords.y);
	//	float y = coords.x*sin(coords.y);
		float x = aperture*rand(seed);
		float y = aperture*rand(seed*x);
		seed = vec2(x, y);
		vec3 n_E = x *U;
		n_E += y * V;
		n_E +=E;
		ray = normalize(focal_point - n_E);
		blur_color += vec3(trace(ray, n_E));

	}
	color = vec4(blur_color/num_blur, 1);
	*/
}