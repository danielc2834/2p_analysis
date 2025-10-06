
# note: run with python 3.8 at least


import os
import re
import xml.etree.ElementTree as ET
from lxml import etree as ET_lxml
import shutil
import numpy as np

def rename_tiff_files(directory, old_cycles, new_cycles):
    for filename in os.listdir(directory):
        if filename.endswith(".ome.tif"):
            for old_cycle, new_cycle in zip(old_cycles, new_cycles):
                old_cycle_str = 'Cycle'+str(old_cycle).zfill(5)  # convert to string and pad with zeros
                new_cycle_str = 'Cycle'+str(new_cycle).zfill(5)  # convert to string and pad with zeros
                if old_cycle_str in filename:
                    new_filename = re.sub(f'{old_cycle_str}', f'{new_cycle_str}', filename)
                    new_filename = 'r_'+ new_filename
                    old_path = os.path.join(directory, filename)
                    new_path = os.path.join(directory, new_filename)
                    os.rename(old_path, new_path)  # rename the file
                    break  # exit the loop once a replacement has been made to avoid double replacements
    for filename in os.listdir(directory):
        if filename.startswith('r_'):
            n_filename=filename.replace('r_','',1)
            new_path = os.path.join(directory, n_filename)
            old_path = os.path.join(directory, filename)
            os.rename(old_path, new_path)

def modify_xml_files(directory, old_cycles, new_cycles):
    old_cycles=sorted(old_cycles,reverse=True)
    new_cycles=sorted(new_cycles,reverse=True)
    
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            tree = ET.parse(os.path.join(directory, filename))  # parse the XML file
            root = tree.getroot()  # get the root element

            for old_cycle, new_cycle in zip(old_cycles, new_cycles):
                old_cycle_str = str(old_cycle)
                new_cycle_str = str(new_cycle)
                
                for sequence in root.findall('.//Sequence'):  # find all Sequence elements in the tree
                    if sequence.attrib.get('cycle') == old_cycle_str:  # if the cycle attribute matches
                        sequence.attrib['cycle'] = new_cycle_str  # replace the cycle attribute value
                        for frame in sequence.findall('Frame'):  # find all Frame elements within this Sequence
                            for file in frame.findall('File'):  # find all File elements within this Frame
                                if '_Cycle' in file.attrib.get('filename', ''):  # if the filename attribute contains _Cycle
                                    file.attrib['filename'] = re.sub(f'_Cycle{old_cycle_str.zfill(5)}_', f'_Cycle{new_cycle_str.zfill(5)}_', file.attrib['filename'])  # replace _CycleXXXXX_ in filename

            # print the XML tree to console before writing to file
            #ET.dump(root)
            # overwrite the original file with the modified XML
            xml_str = ET.tostring(root, encoding='utf8').decode('utf8')

            with open(os.path.join(directory, 'rep_.xml'), 'w', encoding='utf-8') as f:
                f.write(xml_str)


def rewrite_xml_files(dir1_original, dir_repl,cycles_to_include):
    # Find the XML files in the two directories
    xml_file1 = next((f for f in os.listdir(dir1_original) if f.endswith('.xml')), None)
    xml_file2 = next((f for f in os.listdir(dir_repl) if f.endswith('rep_.xml')), None)

    if xml_file1 is None or xml_file2 is None:
        print("XML file not found in one or both directories.")
        return

    # Parse the XML files
    tree1 = ET_lxml.parse(os.path.join(dir1_original, xml_file1))
    tree2 = ET_lxml.parse(os.path.join(dir_repl, xml_file2))

    root1 = tree1.getroot()
    root2 = tree2.getroot()

    # Create a backup directory and copy the original XML file there
    backup_dir = os.path.join(dir1_original, 'backup')
    os.makedirs(backup_dir, exist_ok=True)
    shutil.copy(os.path.join(dir1_original, xml_file1), backup_dir)

    # Find all 'Sequence' elements in the second XML file and store them in a dictionary
    sequences2 = {seq.attrib['cycle']: seq for seq in root2.findall('.//Sequence')}
    sequences1 = {seq.attrib['cycle']: seq for seq in root1.findall('.//Sequence')}
    # Iterate over all 'Sequence' elements in the first XML file
    for seq1 in root1.findall('.//Sequence'):
        cycle = seq1.attrib.get('cycle')
        if cycle in sequences2:
            # Replace the 'Sequence' element and all its children in the first XML file
            parent1 = seq1.getparent()
            index = list(parent1).index(seq1)
            parent1.remove(seq1)
            parent1.insert(index, sequences2[cycle])
            #append
    #for a sequence that was not originally in seq1 but is on seq2
    # find the parent node for 'Sequence' elements in the first XML file
    sequence_parent1 = root1.find('.//Sequence').getparent()
    for seq2 in root2.findall('.//Sequence'):
        cycle = seq2.attrib.get('cycle')
        if cycle not in sequences1 and int(cycle) in cycles_to_include:
            # append the new sequence at the end of the parent node
            sequence_parent1.append(sequences2[cycle])

    # Write the modified XML back to the first file
    tree1.write(os.path.join(dir1_original, xml_file1))


def replace_images(cycles, directory_A, directory_B):
    # Get all .ome.tif files in directory A and B
    files_A = [f for f in os.listdir(directory_A) if f.endswith('.ome.tif')]
    files_B = [f for f in os.listdir(directory_B) if f.endswith('.ome.tif')]

    # Create a dictionary with cycle numbers as keys and file names as values for directory B
    files_B_dict = {}
    cycle_num=[]
    for file in files_B:
        # Assuming the cycle number is always at the same position in the file name
        cycle_num.append(file.split("_")[1][-1])
    cycle_num=np.unique(np.array(cycle_num))    

    # First, loop through all .ome.tif files in directory A and remove the ones with cycle numbers in the list
    for file_A in files_A:
        # Get the cycle number from the file name
        for cycle in cycle_num:
            string_comp="Cycle0000"+ cycle
            if string_comp in file_A:
                # Delete the file in directory A
                os.remove(os.path.join(directory_A, file_A))

    # Then, loop through the dictionary of .ome.tif files in directory B and move the ones with cycle numbers in the list
    for file_B in files_B:
        # Move the files from directory B to directory A
        shutil.move(os.path.join(directory_B, file_B), 
                        os.path.join(directory_A, file_B))

def eliminate_cycles(cycle_num,dir_original_TSER):
    files_A = [f for f in os.listdir(dir_original_TSER) if f.endswith('.ome.tif')]
    for file_A in files_A:
    # Get the cycle number from the file name
        for cycle in cycle_num:
            string_comp="Cycle0000"+ str(cycle)
            if string_comp in file_A:
                # Delete the file in directory A
                os.remove(os.path.join(dir_original_TSER, file_A))

#%%
directory_replacement=r'C:\Users\vargasju\PhD\experiments\2p\T4T5_STRF_glucla_rescues\raw\20240124_jv_fly1\TSeries-008\23'
dir_original_TSER=r'C:\Users\vargasju\PhD\experiments\2p\T4T5_STRF_glucla_rescues\raw\20240124_jv_fly1\TSeries-008'
#eliminate_cycles([2],directory_replacement)
#rename_tiff_files(directory_replacement,[1,2],[2,3])#,2,3,4,5   2,3,4,5,
#modify_xml_files(directory_replacement,[1,2],[2,3])#,2,3,4,5
#rewrite_xml_files(dir_original_TSER, directory_replacement,[2,3])#[2,3,4,5,6]
#replace_images([2,3], dir_original_TSER, directory_replacement)

os.chdir(dir_original_TSER)

for tser in os.listdir():
    if 'TSeries-008' in tser:
        new_filename = re.sub('TSeries-008', 'TSeries-007', tser)
        os.rename(tser, new_filename)