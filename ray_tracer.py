import glfw
from glfw import *
import pyglet
from pyglet.gl import *
from pyshaders.pyshaders import from_files_names, ShaderCompilationError,ShaderUniformAccessor
from pyglbuffers.pyglbuffers import *
import numpy as np
from load_obj import load_obj
from aabbtree import AABB, AABBTree
from ctypes import *
from datetime import datetime as dt


class struct:
    def __init__(self, d):
        for k in d:
            setattr(self, k, d[k])

def attach_uniform_buffer(buff, shader, name, binding_point):
    block_index = (glGetUniformBlockIndex(shader.pid,  name))    
    glUniformBlockBinding(shader.pid, block_index, binding_point)
    glBindBufferBase(GL_UNIFORM_BUFFER, binding_point, buff.bid)

def normalize(a, axis = 0, keepdims = True):
    norm = np.sqrt((a*a).sum(axis = axis, keepdims = keepdims))
    return a / norm
class struct:
    def __init__(self, d):
        for k in d:
            setattr(self, k, d[k])

def window_size_callback(window_dict, window, width, height):
    window_dict.width = width
    window_dict.height = height
    glViewport(0, 0, width, height)

def cursor_position_callback(window_dict, window, xpos, ypos):
    if not window_dict.show_cursor:
        m_pos = np.array([xpos, ypos])
        if window_dict.mouse is None:
            delta = np.zeros(2)
        else:
            delta = m_pos - window_dict.mouse 
        delta *= np.array([1, -1])
        sensitivity = 0.05* window_dict.delta_t
        delta *= sensitivity

        window_dict.yaw -= delta[0]
        window_dict.pitch += delta[1]

        if window_dict.pitch > np.pi /2 - 0.01:
            window_dict.pitch = np.pi/2 - 0.01
        if window_dict.pitch < -np.pi /2 + 0.01:
            window_dict.pitch = -np.pi/2 + 0.01
        front = np.array([
            np.sin(window_dict.yaw)*np.cos(window_dict.pitch),
            np.sin(window_dict.pitch),
            np.cos(window_dict.yaw)*np.cos(window_dict.pitch)])
        window_dict.W = normalize(front)
        window_dict.mouse = m_pos

def set_cursor(window, window_dict):
    if window_dict.show_cursor:
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)
    else:
        window_dict.mouse = None
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

def key_callback(window_dict, window, key, scancode, action, mods):
    
    if key == glfw.KEY_W:
        if action == glfw.PRESS:
            window_dict.up = True
        elif action == glfw.RELEASE:
            window_dict.up = False
    if key == glfw.KEY_A:
        if action == glfw.PRESS:
            window_dict.left = True
        elif action == glfw.RELEASE:
            window_dict.left = False
    if key == glfw.KEY_D: 
        if action == glfw.PRESS:
            window_dict.right = True
        elif action == glfw.RELEASE:
            window_dict.right = False
    if key == glfw.KEY_S :
        if action == glfw.PRESS:
            window_dict.down = True
        elif action == glfw.RELEASE:
            window_dict.down = False
    if key == glfw.KEY_LEFT_SHIFT:
        if action == glfw.PRESS:
            window_dict.sprint = True

        elif action == glfw.RELEASE:
            window_dict.sprint = False

    if key == glfw.KEY_ESCAPE and action == glfw.RELEASE:
        window_dict.show_cursor = 1 - window_dict.show_cursor
        set_cursor(window, window_dict)

def rot_y(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]])

def move_cam(window_dict):
    direction = normalize(window_dict.W * np.array([1, 0, 1]))
    rot_90 = rot_y(np.pi / 2)
    direction_rot = np.matmul(rot_90, direction)
    sensitivity = 3.5*window_dict.delta_t
    if window_dict.sprint:
        sensitivity *= 3
    if window_dict.up:
        window_dict.E += sensitivity * direction
    if window_dict.down:
        window_dict.E -= sensitivity * direction
    if window_dict.left:
        window_dict.E += sensitivity * direction_rot
    if window_dict.right:
        window_dict.E -= sensitivity * direction_rot

def f(tr, tree, i):
    
    min_vals = np.min(tr, axis = 0)
    max_vals = np.max(tr, axis = 0)
    
    extremes = np.array([min_vals, max_vals]).T + np.array([-1, 1])*0.1
    aabb = AABB([tuple(x) for x in extremes])
    tree.add(aabb, i)

def get_obj_data(file, tree, num_obj):



    vertices, faces = load_obj(file, normalization=False)

    triangles = vertices[faces] #+ np.array([0, 5, 0])
    i = num_obj
    for tr in triangles:
        f(tr, tree, i)
        i +=1
    tr_base = np.array(triangles[:, 0:1, :])
    tr_vecs = np.array(triangles[:, 1:3:, :])
    tr_vecs -= tr_base
    normals = normalize(np.cross(tr_vecs[:, 0, :], tr_vecs[:, 1, :]), axis = 1)
   
    triangles = triangles.reshape(-1, 9)
    
    data = np.append(triangles, normals, axis = 1)
    data = np.append(data, normals, axis = 1)
    data = data.reshape(-1, 5, 3)
    return data

def mouse_button_callback(window_dict, window, button, action, mods):
    if glfw.PRESS:
        window_dict.show_cursor = False
        set_cursor(window, window_dict)

def traverse(tree, f):
    if(tree is None):
        return
    f(tree)
    traverse(tree.left, f)
    traverse(tree.right, f)

def init():
    glfw.init()
    m = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(m)
    width = mode.size.width //4
    height = mode.size.height //4
    window = glfw.create_window(width, height, 'Ray Tracer', None, None)
    glfw.make_context_current(window)

    window_dict = {
        'width': width,
        'height': height,
        'W' : np.array([0, 0, 1]),
        'E' : np.array([0, 0.15, -1]),
        'b' : np.array([0, 1, 0]),
        'd' : 1.5,
        'mouse' : None,
        'yaw' : 0,
        'pitch' : 0,
        'up' : False,
        'down' : False,
        'left' : False,
        'right' : False,
        'show_cursor' : False,
        'num_triangles' : 0,
        'num_objects' : 2,
        'fov' : 90,
        'delta_t' : 0,
        'sprint' : False,
        'num_aabb': 0
    }
    window_dict = struct(window_dict)
    glfw.set_window_size_callback(window, lambda *args : window_size_callback(window_dict, *args))
    glfw.set_cursor_pos_callback(window, lambda *args : cursor_position_callback(window_dict, *args))
    glfw.set_key_callback(window, lambda *args : key_callback(window_dict, *args))
    glfw.set_mouse_button_callback(window, lambda *args : mouse_button_callback(window_dict,*args))
    set_cursor(window, window_dict)
    
    quad = pyglet.graphics.vertex_list(4,
        ('v2f', (-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0)))
   
    try:
        shader = from_files_names("quad_vshader.glsl", "quad_fshader.glsl")
    except ShaderCompilationError as e:
        print(e.logs) 
        exit()
    

    GL_SHADER_STORAGE_BUFFER = GLuint(0x90D2)
    
    f_obj = '(1f)[type](3f)[pos](1f)[size1](1f)[size2](3f)[color](3f)[direction]'
    object_buffer = Buffer.array(format = f_obj, usage = GL_DYNAMIC_DRAW)
    object_buffer.bind(GL_SHADER_STORAGE_BUFFER)
    #object_buffer.reserve(3)
    #attach_uniform_buffer(object_buffer, shader, b'object_data', GLuint(1))
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, object_buffer.bid)
    

    f_tr = '(3f)[vertex1](3f)[vertex2](3f)[vertex3](3f)[color](3f)[normal]'
    tr_buffer = Buffer.array(format = f_tr,usage = GL_DYNAMIC_DRAW)
    tr_buffer.bind(GL_SHADER_STORAGE_BUFFER)
    #attach_uniform_buffer(tr_buffer, shader, b'triangle_data', GLuint(2))
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, tr_buffer.bid)
   
    #f_aabb = '(2f)[xlim](2f)[ylim](2f)[zlim](1i)[left](1i)[right]'

    
    
    tree = AABBTree()

    
    data = get_obj_data('bunny.obj', tree, window_dict.num_objects)
    print(tree.depth)
    aabb = AABB([(-1, 1), (-1, 1), (9, 11)])
    tree.add(aabb, 0)
    aabb = AABB([(-15, 15), (-1.1, 0.9), (-15, 15) ])
    tree.add(aabb, 1)
   # print(tree)
    
    tree_arr = []
    def f(tree_arr, tree):
        left = tree.left.num if tree.left else -1
        right = tree.right.num if tree.right else tree.value
        limits = [list(x) for x in tree.aabb.limits]
        l = [[2], limits[0] + [limits[1][0]]   , [left], [right], [0, 0, 0], [limits[1][1]] + limits[2] ]
        tree_arr += [l]
    def g(tree, num):
        tree.num = num['num']
        num['num'] += 1
        
    num = {'num' : 0}
    traverse(tree, lambda x : g(x, num))
    traverse(tree, lambda x : f(tree_arr, x))



    window_dict.num_aabb = len(tree_arr)
    object_data = []
    object_data += [[[0], [0, 0, 10], [1], [0], [1, 0, 0], [0, 0, 0]]]
    object_data += [[[1], [0, -1, 1], [0], [0],[1, 0, 1] ,[0, 1, 0]]]
    object_data += tree_arr

    object_buffer.init(object_data)
    print(tree.depth)


    #for e, i in zip(object_data, range(len(object_data))):
    #    print(i, e)
    
    #aabb_buf.init(tree_arr)
    
   # aabb_buf.reserve(1)
   # aabb_buf[0] = [[1, 0], [0, 0], [0, 0], [0], [0]] 
    

    window_dict.num_triangles = len(data)
    tr_buffer.init(data)

    return window, window_dict, object_buffer, tr_buffer, shader, quad

def main():
    


    window, window_dict, object_buffer, tr_buffer, shader, quad = init()

    shader.use()
   # AR = window_dict.width / window_dict.height
    print(shader.uniforms)
    shader.uniforms.top = 1
    shader.uniforms.bottom = -1
    shader.uniforms.num_triangles = window_dict.num_triangles
    shader.uniforms.light_pos = [[0, 10, 5]]
    shader.uniforms.num_lights = 1
    shader.uniforms.num_objects = 2
    try:
        shader.uniforms.num_aabb = window_dict.num_aabb
    except AttributeError:
        pass
    start = dt.now()
    while not glfw.window_should_close(window):
        end = dt.now()
        del_t = (end - start)
        window_dict.delta_t = del_t.microseconds*1e-6 + del_t.seconds
        start = end
       # print(window_dict.delta_t)
        move_cam(window_dict)
        AR = window_dict.width / window_dict.height
      
        window_dict.d = AR / np.tan(np.pi*window_dict.fov / 360)
        U = normalize(np.cross(window_dict.W, window_dict.b))
        V = normalize(np.cross(U, window_dict.W))
        S = window_dict.E + window_dict.d*window_dict.W
        
        shader.uniforms.U = tuple(U)
        shader.uniforms.V = tuple(V)
        shader.uniforms.S = tuple(S)
        shader.uniforms.E = tuple(window_dict.E)
        shader.uniforms.left = -AR
        shader.uniforms.right = AR

        glfw.swap_buffers(window)
        glfw.poll_events()
        quad.draw(GL_TRIANGLE_STRIP)

if __name__ == "__main__":
    main()