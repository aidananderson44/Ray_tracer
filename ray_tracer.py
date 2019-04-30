import glfw
from glfw import *
import pyglet
from pyglet.gl import *
from pyshaders.pyshaders import from_files_names, ShaderCompilationError
from pyglbuffers.pyglbuffers import *
import numpy as np
from load_obj import load_obj
class struct:
    def __init__(self, d):
        for k in d:
            setattr(self, k, d[k])

def attach_uniform_buffer(buff, shader, name, binding_point):
    block_index = (glGetUniformBlockIndex(shader.pid,  name))
    glUniformBlockBinding(shader.pid, block_index, binding_point)
    glBindBufferBase(GL_UNIFORM_BUFFER, binding_point, buff.bid)

def normalize(a, axis = 0):
    norm = np.sqrt((a*a).sum(axis = axis, keepdims = True))
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
        sensitivity = 0.005
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

def set_cursor(window, show):
    if show:
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)
    else:
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
    if key == glfw.KEY_ESCAPE and action == glfw.RELEASE:
        window_dict.show_cursor = 1 - window_dict.show_cursor
        set_cursor(window, window_dict.show_cursor)

def rot_y(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]])

def move_cam(window_dict):
    direction = normalize(window_dict.W * np.array([1, 0, 1]))
    rot_90 = rot_y(np.pi / 2)
    direction_rot = np.matmul(rot_90, direction)
    sensitivity = 0.1
    if window_dict.up:
        window_dict.E += sensitivity * direction
    if window_dict.down:
        window_dict.E -= sensitivity * direction
    if window_dict.left:
        window_dict.E += sensitivity * direction_rot
    if window_dict.right:
        window_dict.E -= sensitivity * direction_rot

def get_obj_data(file):
    vertices, faces = load_obj(file, normalization=False)

    triangles = vertices[faces] #+ np.array([0, 5, 0])
    tr_base = np.array(triangles[:, 0:1, :])
    tr_vecs = np.array(triangles[:, 1:3:, :])
    tr_vecs -= tr_base
    normals = normalize(np.cross(tr_vecs[:, 0, :], tr_vecs[:, 1, :]), axis = 1)
   
    triangles = triangles.reshape(-1, 9)
    
    data = np.append(triangles, normals, axis = 1)
    data = np.append(data, normals, axis = 1)
    data = data.reshape(-1, 5, 3)
    return data

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
        'E' : np.array([0, 0.25, -1]),
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
        'num_objects' : 2
    }
    window_dict = struct(window_dict)
    glfw.set_window_size_callback(window, lambda *args : window_size_callback(window_dict, *args))
    glfw.set_cursor_pos_callback(window, lambda *args : cursor_position_callback(window_dict, *args))
    glfw.set_key_callback(window, lambda *args : key_callback(window_dict, *args))
    set_cursor(window, window_dict.show_cursor)
    
    quad = pyglet.graphics.vertex_list(4,
        ('v2f', (-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0)))
   
    try:
        shader = from_files_names("quad_vshader.glsl", "quad_fshader.glsl")
    except ShaderCompilationError as e:
        print(e.logs) 
        exit()
    s_pid = shader.pid
    f_obj = '(1f)[type](3f)[pos](1f)[size1](1f)[size2](3f)[color](3f)[direction]'
    object_buffer = Buffer.array(format = f_obj,usage = GL_DYNAMIC_DRAW)
    object_buffer.bind(GL_UNIFORM_BUFFER)
    object_buffer.reserve(10)
    attach_uniform_buffer(object_buffer, shader, b'object_data', GLuint(1))

    f_tr = '(3f)[vertex1](3f)[vertex2](3f)[vertex3](3f)[color](3f)[normal](1f)[padding]'
    tr_buffer = Buffer.array(format = f_tr,usage = GL_DYNAMIC_DRAW)
    tr_buffer.bind(GL_UNIFORM_BUFFER)
    
    attach_uniform_buffer(tr_buffer, shader, b'triangle_data', GLuint(2))
  
    object_buffer[0] = [[0], [0, 0, 10], [1], [0], [1, 0, 0], [0, 0, 0]]
    object_buffer[1] = [[1], [0, -1, 1], [0], [0],[1, 0, 1] ,[0, 1, 0]]

    data = get_obj_data('aquacisi.obj')
    window_dict.num_triangles = len(data)
    tr_buffer.init(data)

    return window, window_dict, object_buffer, tr_buffer, shader, quad

def main():
    
    window, window_dict, object_buffer, tr_buffer, shader, quad = init()
    shader.use()

    
    while not glfw.window_should_close(window):
        move_cam(window_dict)
        AR = window_dict.width / window_dict.height

        U = normalize(np.cross(window_dict.W, window_dict.b))
        V = normalize(np.cross(U, window_dict.W))
        S = window_dict.E + window_dict.d*window_dict.W
        
        shader.uniforms.num_triangles = window_dict.num_triangles
        shader.uniforms.light_pos = [[0, 10, 5]]
        shader.uniforms.num_lights = 1
        shader.uniforms.num_objects = 2
        shader.uniforms.U = tuple(U)
        shader.uniforms.V = tuple(V)
        shader.uniforms.S = tuple(S)
        shader.uniforms.E = tuple(window_dict.E)
        shader.uniforms.left = -AR
        shader.uniforms.right = AR
        shader.uniforms.top = 1
        shader.uniforms.bottom = -1

        glfw.swap_buffers(window)
        glfw.poll_events()
        quad.draw(GL_TRIANGLE_STRIP)

if __name__ == "__main__":
    main()