import bpy  
import bmesh  
import os  
from mathutils import Vector  
import math  
import time  
from datetime import datetime  

# Parameters  
stl_file_root = "sim/assets/indenters/output/stl"  
texture_file_path = "sim/marker/marker_pattern"  
output_image_path = "sim/assets/marker"  

def clean_up():  
    bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects  
    for obj in bpy.data.objects:  
        if obj.type in {'MESH', 'LIGHT', 'CAMERA'}:  
            bpy.data.objects.remove(obj, do_unlink=True)  

def load_image_texture(texture_file_path):  
    # 3. Load the image texture  
    image = bpy.data.images.load(texture_file_path)   
    # 4. Create a material and assign the texture  
    material = bpy.data.materials.new(name="TextureMaterial")  
    material.use_nodes = True  
    bsdf = material.node_tree.nodes.get('Principled BSDF')  
    # Create an image texture node  
    tex_image = material.node_tree.nodes.new('ShaderNodeTexImage')  
    tex_image.image = image  
    # Add texture coordinate node to use UV mapping  
    tex_coord = material.node_tree.nodes.new('ShaderNodeTexCoord')  
    # Link UV coordinates directly to the texture  
    material.node_tree.links.new(tex_coord.outputs['UV'], tex_image.inputs['Vector'])  
    # Link the image texture to the base color of the BSDF  
    material.node_tree.links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])  # 6. Set the material specular to 0  
    if bsdf:  
        bsdf.inputs["Specular"].default_value = 0.0   

    return material  

def set_world():  
    # 7. Set camera location and rotation  
    camera_loc = (19.955, 19.576, -9.1104)   
    camera_rot = (math.radians(-180), 0, 0)  

    # Position the camera  
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))  
    camera.location = camera_loc  # Adjust position  
    camera.rotation_euler = camera_rot  # Pointing down at the object  

    # Set the camera focal length  
    camera.data.lens = 35  # Focal length in mm  

    # Link the camera to the scene  
    bpy.context.collection.objects.link(camera)  
    bpy.context.scene.camera = camera  

    # 8. Set the world light strength  
    bpy.context.scene.world.use_nodes = True  
    world = bpy.context.scene.world  
    bg = world.node_tree.nodes.get('Background')  
    bg.inputs['Strength'].default_value = 60.0  # Set the world light strength  

    # 9. Set the render output settings  
    bpy.context.scene.render.image_settings.file_format = 'JPEG'  
    bpy.context.scene.render.resolution_x = 640  
    bpy.context.scene.render.resolution_y = 480  
    bpy.context.scene.render.resolution_percentage = 100   

def check_and_rotate_uvs(obj):  
    # Get the mesh data  
    me = obj.data  
    bm = bmesh.from_edit_mesh(me)  
    
    # Get UV layer  
    uv_layer = bm.loops.layers.uv.verify()  
    
    # Get object bounds in local space  
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]  
    xmin = min(corner.x for corner in bbox_corners)  
    xmax = max(corner.x for corner in bbox_corners)  
    ymin = min(corner.y for corner in bbox_corners)  
    ymax = max(corner.y for corner in bbox_corners)  
    
    # Try up to 4 rotations (0°, 90°, 180°, 270°)  
    for rotation_count in range(4):  
        # Find UV coordinates at xmax,ymax and xmin,ymin  
        uv_at_max = None  
        uv_at_min = None  
        for face in bm.faces:  
            if not face.select:  
                continue  
            for loop in face.loops:  
                vert = loop.vert  
                uv_coords = loop[uv_layer]  
                if abs(vert.co.x - xmax) < 0.001 and abs(vert.co.y - ymax) < 0.001:  
                    uv_at_max = Vector((uv_coords.uv.x, uv_coords.uv.y))  
                if abs(vert.co.x - xmin) < 0.001 and abs(vert.co.y - ymin) < 0.001:  
                    uv_at_min = Vector((uv_coords.uv.x, uv_coords.uv.y))  
        
        if uv_at_max and uv_at_min:  
            # Check if conditions are met  
            if (abs(uv_at_max.x -1) < 0.1 and abs(uv_at_max.y - 0) < 0.1 and  
                abs(uv_at_min.x - 0) < 0.1 and abs(uv_at_min.y - 1) < 0.1):  
                # Conditions met, exit the function  
                print(f"UV mapping correct after {rotation_count} rotations")  
                break
            else:  
                # Rotate UV coordinates 90 degrees clockwise  
                for face in bm.faces:  
                    if not face.select:  
                        continue  
                    for loop in face.loops:  
                        uv_coords = loop[uv_layer]  
                        old_x, old_y = uv_coords.uv.x, uv_coords.uv.y  
                        uv_coords.uv.x = old_y  
                        uv_coords.uv.y =1 - old_x  
                # Update the mesh after rotation  
                bmesh.update_edit_mesh(me)
                print(f"Rotated UVs {rotation_count + 1} times")  
        if rotation_count == 3:  
            print("Warning: Could not achieve correct UV orientation after all rotations")  
    
    bmesh.update_edit_mesh(me)  

def render(material, stl_file_path, output_image_path):  
    # 1. Clean up existing objects, camera, and lights      
    clean_up()  
    set_world()  
    # 2. Import the STL file  
    bpy.ops.import_mesh.stl(filepath=stl_file_path)  
    # Get the active object (which should be the imported STL)  
    obj = bpy.context.active_object  
    # Assign material to the object  
    if obj.data.materials:  
        obj.data.materials[0] = material  
    else:  
        obj.data.materials.append(material)  
    # 5. Set UV unwrapping for the surface with negative Z-axis normals  
    bpy.ops.object.mode_set(mode='EDIT')  
    bm = bmesh.from_edit_mesh(obj.data)  
    # Select faces with negative Z normal  
    for face in bm.faces:  
        if face.normal.z < 0:  
            face.select = True  
    
    # Unwrap the selected faces  
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)  
    
    # Check and rotate UVs if needed  
    check_and_rotate_uvs(obj)  
    
    # Return to object mode  
    bpy.ops.object.mode_set(mode='OBJECT')   

    # 10. Render the scene and save the image  
    bpy.context.scene.render.filepath = output_image_path  
    bpy.ops.render.render(write_still=True)  

def main():  

    target_marker = ["Array1", "Array2", "Array3", "Array4",\
                     "Circle1","Circle2","Circle3","Circle4",
                     "Diamond1","Diamond2","Diamond3","Diamond4"]  
    
    target_indenter = ["cone", "curface", "cylinder_sh", "cylinder_si", "cylinder",\
                       "dot_in", "dots", "hexagon","line","moon","pacman","prism",\
                        "random", "sphere_s", "sphere", "torus", "triangle", "wave"]    # 18 types of indenters
    t_now = time.time()  
    dt_obj = datetime.fromtimestamp(t_now)  
    timestamp_str = dt_obj.strftime("%m_%d_%H_%M")  
    for indenter in target_indenter:  
        stl_file_path = os.path.join(stl_file_root, indenter)
        if not os.path.exists(stl_file_path):
            print(f"{indenter} not found")
            continue
        stls = os.listdir(stl_file_path)  
        root_dir = os.path.join(output_image_path, timestamp_str, indenter)  
        if not os.path.exists(root_dir):  
            os.makedirs(root_dir)  
        for texture in target_marker:  
            if not os.path.exists(os.path.join(root_dir, texture)):  
                os.makedirs(os.path.join(root_dir, texture))  
            texture_dir = os.path.join(texture_file_path, texture + ".jpg")  
            material = load_image_texture(texture_dir)  
            for stl in stls:  
                out_file_name = stl[:-4] + ".jpg"  
                stl_dir = os.path.join(stl_file_path, stl)  
                output_dir = os.path.join(root_dir, texture, out_file_name)  
                render(material, stl_dir, output_dir)  

if __name__ == "__main__":  
    main()