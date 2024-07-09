"""
Convert musical symbols and pitches to MEI and optionally Humdrum format.

Parameters:
    symbols_and_pitches : dict
        A dictionary containing the musical symbols and their 
        corresponding pitches.
    humdrum : bool, optional
        If True, converts to both MEI and Humdrum formats. If False, 
        converts to MEI format only. Default is True.

Returns:
    None
    The function does not return any value. It saves the converted 
    MEI and Humdrum files to disk.

Utility Modules:
    mensural_to_mei.utils: Provides utility functions for converting
        symbols and pitches to MEI format.
    colorama: Cross-platform colored terminal text.
    termcolor: ANSI color formatting for output in terminal.

Example:
    >>> symbols_and_pitches = {
    ...     'symbols': [{'type': 'clef', 'pitch': 'c-c-g'}, 
    ...                 {'type': 'flat'}, 
    ...                 {'type': 'mens', 'pitch': 'met_3_2'}, 
    ...                 {'type': 'dot'}, 
    ...                 {'type': 'note', 'pitch': 'a4'}],
    ...     'metadata': ['example']
    ... }
    >>> convert_to_mei_and_humdrum(symbols_and_pitches)
    Converting to MEI and Humdrum
    Conversion complete!

Note:
    The function generates unique IDs for MEI elements, sets up initial 
    MEI and Humdrum declarations, and iterates over the provided musical 
    symbols and pitches to construct the MEI and Humdrum representations. 
    It handles various musical elements such as clefs, key signatures, 
    mensurations, notes, rests, dots, barlines, and custos. The resulting 
    files are saved incrementally based on the provided metadata.
"""

import os
import sys

from tqdm import tqdm
sys.path.append('../mensural_to_mei')

from datetime import datetime
import pickle
from mensural_to_mei.utils import convert_to_combined_list_with_metadata, generate_random_numbers, prettyprint
from lxml import etree
from colorama import just_fix_windows_console
from termcolor import cprint

just_fix_windows_console()

# global counters
id_counter = 0
staff_counter = 0
layer_counter = 0
used_filenames = []
VERSION = '1.0.0'


def increment_staffcounter():
    global staff_counter
    staff_counter += 1
    return staff_counter


def increment_idcounter():
    global id_counter
    id_counter += 1
    return id_counter


def increment_layercounter():
    global layer_counter
    layer_counter += 1
    return layer_counter


def convert_to_mei_and_humdrum(
        symbols_and_pitches: dict,
        humdrum: bool=True) -> None:
    
    """
    Convert musical symbols and pitches to MEI and optionally Humdrum format.

    Parameters
    ----------
    symbols_and_pitches : dict
        A dictionary containing the musical symbols and their 
        corresponding pitches.
    humdrum : bool, optional
        If True, converts to both MEI and Humdrum formats. If False, 
        converts to MEI format only. Default is True.

    Returns
    -------
    None
        The function does not return any value. It saves the converted 
        MEI and Humdrum files to disk.

    Notes
    -----
    The function generates unique IDs for MEI elements, sets up initial 
    MEI and Humdrum declarations, and iterates over the provided musical 
    symbols and pitches to construct the MEI and Humdrum representations. 
    It handles various musical elements such as clefs, key signatures, 
    mensurations, notes, rests, dots, barlines, and custos. The resulting 
    files are saved incrementally based on the provided metadata.

    Examples
    --------
    >>> symbols_and_pitches = {
    ...     'symbols': [{'type': 'clef', 'pitch': 'c-c-g'}, 
    ...                 {'type': 'flat'}, 
    ...                 {'type': 'mens', 'pitch': 'met_3_2'}, 
    ...                 {'type': 'dot'}, 
    ...                 {'type': 'note', 'pitch': 'a4'}],
    ...     'metadata': ['example']
    ... }
    >>> convert_to_mei_and_humdrum(symbols_and_pitches)
    Converting to MEI and Humdrum
    Conversion complete!
    """

    ids = generate_random_numbers(10000)

    if not humdrum:
        cprint('Converting to MEI', 'blue')
    else:
        cprint('Converting to MEI and Humdrum', 'blue')

    RESTS_MEI = {
        'r-lo': 'longa',
        'r-br': 'brevis',
        'r-sb': 'semibrevis',
        'r-mi': 'minima',
        'r-sm': 'semiminima',
        'r-fu': 'fusa',
        'r-se': 'semifusa',
    }

    NOTES_MEI = {
        'ma': 'maxima',
        'lo': 'longa',
        'bre': 'brevis',
        'sebre': 'semibrevis',
        'mi': 'minima',
        'sm': 'semiminima',
        'fu': 'fusa',
        'sf': 'semifusa',
        'br': 'brevis',
        'sb': 'semibrevis',
        'li': 'semibrevis',
    }

    NOTES_HUMDRUM = {
        'ma': 'X',
        'lo': 'L',
        'bre': 'S',
        'sebre': 's',
        'mi': 'M',
        'sm': 'm',
        'fu': 'U',
        'sf': 'u',
        'br': 'S~',
        'sb': 's~',
        'li': '[s'
    }

    RESTS_HUMDRUM = {
        'r-lo': 'Lr',
        'r-br': 'Sr',
        'r-sb': 'sr',
        'r-mi': 'Mr',
        'r-sm': 'mr',
        'r-fu': 'Ur',
        'r-se': 'ur',
    }

    CLEFS_HUMDRUM = {
        "c-c-g": "*clefC2",
        "c-c-e": "*clefC1",
        "c-c-b": "*clefC3",
        "c-c-d": "*clefC4",
        "c-g": "*clefG2",
        "c-f-b": "*clefF3",
        "c-f": "*clefF4"
    }


    stafflist, metadata = convert_to_combined_list_with_metadata(symbols_and_pitches)

    filename_counter = 0
    humdrum_string = ['**mens']
    for ix_staff, staff in enumerate(tqdm(stafflist, desc="Converting")):
        saved = False
        if ix_staff == 0:
            mei = etree.Element('mei')
            mei.set('xmlns', 'http://www.music-encoding.org/ns/mei')
            mei.set('meiversion', '5.0')

            mei.append(create_meihead())

            xml_clef, xml_keySign, xml_layer = create_mei_declarations(ids, mei)

            filename = metadata[0]

            first_staff = True

        if first_staff:
            first_staff = False
            initial_flat_found = False
            # check if first element in first staff is clef
            if staff[0]['type'] == 'clef':
                initial_clef = staff[0]
                if staff[1]['type'] == 'flat':
                    initial_flat_found = True
            else:
                # no clef found in the first staff, check if clef in subsequent staff
                for i in range(1, len(stafflist)-1):
                    if stafflist[i][0]['type'] == 'clef':
                        initial_clef = stafflist[i][0]
                        if stafflist[i][1]['type'] == 'flat':
                            initial_flat_found = True

                        break

            if initial_clef is None:
                # no clef found in the first staff and in the subsequent staffs
                cprint('No clef was found in the staffs. Clefs are necessary to perform conversion!', 'red')
                sys.exit('No clef was found in the staffs. Clefs are necessary to perform conversion!')
            
            clef = initial_clef        
            initial_key = ''
            shape, line = analyse_clef(initial_clef)
            xml_clef.set('{http://www.w3.org/XML/1998/namespace}id',
                            f'clef-{ids[increment_idcounter()]}')
            xml_clef.set('shape', shape)
            xml_clef.set('line', line)

            if humdrum:
                humdrum_string.append(CLEFS_HUMDRUM[initial_clef['pitch']])

            if initial_flat_found:
                xml_keySign.set(
                    '{http://www.w3.org/XML/1998/namespace}id', f'keySig-{ids[increment_idcounter()]}')
                xml_keySign.set('sig', '1f')
                initial_key = 'flat'

                if humdrum:
                    humdrum_string.append('*k[b-]')
        
        sharp_found = False
        flat_found = False
        ligature = False
        for ix_symbol, symbol in enumerate(staff):
            t = symbol['type']
            if symbol == initial_clef:
                clef = initial_clef
                continue

            if t == initial_key and ix_symbol == 1:
                continue

            if t == initial_key and ix_symbol == 2:
                if clef['pitch'] == 'c-c-g':
                    continue

            if t == 'flat':
                flat_found = True
                continue

            if t == 'sharp':
                sharp_found = True
                continue

            if t == 'mens':
                if symbol['pitch'] == 'met_3_2':
                    mensur.set('num', '3')
                    mensur.set('numbase', '2')
                    if humdrum:
                        humdrum_string[-1] = humdrum_string[-1][:-1] + '3/2)'
                        
                    continue
                
                mensur = etree.SubElement(xml_layer, 'mensur')
                mensur.set('{http://www.w3.org/XML/1998/namespace}id',
                            f'mens-{ids[increment_idcounter()]}')

                sign, slash = analyse_mensuration(symbol)
                mensur.set('prolatio', '2')
                mensur.set('sign', sign)
                if slash:
                    mensur.set('slash', slash)

                if humdrum:
                    sl = ''
                    if slash:
                        sl = '|'

                    humdrum_line = f'*met({sign}{sl})'

            if t == 'dot':
                dot = etree.SubElement(xml_layer, 'dot')
                dot.set('{http://www.w3.org/XML/1998/namespace}id',
                        f'dot-{ids[increment_idcounter()]}')
                
                if humdrum:
                    humdrum_string[-1] = humdrum_string[-1][0] + ':' + humdrum_string[-1][1:]
                    continue

            if t in RESTS_MEI.keys():
                rest = etree.SubElement(xml_layer, 'rest')
                rest.set('{http://www.w3.org/XML/1998/namespace}id',
                            f'rest-{ids[increment_idcounter()]}')
                rest.set('dur', RESTS_MEI[t])

                if humdrum:
                    humdrum_line = RESTS_HUMDRUM[t]

            if t in NOTES_MEI.keys():
                oct, pname = analyse_note(clef, symbol['pitch'])
                note = etree.SubElement(xml_layer, 'note')
                note.set('{http://www.w3.org/XML/1998/namespace}id',
                            f'note-{ids[increment_idcounter()]}')
                note.set('dur', NOTES_MEI[t])
                note.set('oct', str(oct))
                note.set('pname', pname)

                if humdrum and not ligature:
                    line_humdrum = NOTES_HUMDRUM[t]
                
                if t == 'br' or t == 'sb':
                    note.set('colored', 'true')
                    if humdrum:
                        line_humdrum = NOTES_HUMDRUM[t]

                if humdrum:
                    line_humdrum = get_humdrum_pitch(oct, pname, line_humdrum)

                if t == 'li':
                    xml_lig = etree.SubElement(xml_layer, 'ligature')
                    xml_lig.set('{http://www.w3.org/XML/1998/namespace}id',
                                f'ligature-{ids[increment_idcounter()]}')
                    xml_lig.set('form', 'recta')
                    xml_lig.append(note)
                    ligature = True
                    if humdrum:
                        line_humdrum = '[s'
                        line_humdrum = get_humdrum_pitch(oct, pname, line_humdrum)
                        humdrum_string.append(line_humdrum)
                    continue

                if ligature:
                    note.attrib['dur'] = 'semibrevis'
                    xml_lig.append(note)
                    ligature = False
                    if humdrum:
                        line_humdrum = 's'
                        line_humdrum = get_humdrum_pitch(oct, pname, line_humdrum)
                        line_humdrum += ']'
                        humdrum_line = line_humdrum

                if sharp_found:
                    note.set('accid', 's')
                    sharp_found = False
                    if humdrum:
                        line_humdrum += '#'

                if flat_found:
                    note.set('accid', 'f')
                    flat_found = False
                    if humdrum:
                        line_humdrum += '-'
                        
                if humdrum:
                    humdrum_line = line_humdrum     

            if t == 'bar':
                barline = etree.SubElement(xml_layer, 'barLine')
                barline.set('{http://www.w3.org/XML/1998/namespace}id',
                            f'barline-{ids[increment_idcounter()]}')
                barline.set('form', 'dbl')

                if humdrum:
                    humdrum_string.append('=||')
                    humdrum_string.append('*-')

                filename, filename_counter = save_mei_file(metadata, filename_counter, ix_staff, mei, filename, stafflist, humdrum_string,
                                                           humdrum)

                del mei

                mei = etree.Element('mei')
                mei.set('xmlns', 'http://www.music-encoding.org/ns/mei')
                mei.set('meiversion', '5.0')

                mei.append(create_meihead())

                xml_clef, xml_keySign, xml_layer = create_mei_declarations(ids, mei)

                first_staff = True
                saved = True
                continue

            if t == 'custos':
                custos = etree.SubElement(xml_layer, 'custos')
                custos.set('{http://www.w3.org/XML/1998/namespace}id',
                            f'custos-{ids[increment_idcounter()]}')
                if humdrum:
                    line_humdrum = '*custos'

                if ix_staff < len(stafflist) - 1:
                    for symbol in stafflist[ix_staff + 1]:
                        if symbol['type'] in NOTES_MEI.keys():
                            oct, pname = analyse_note(clef, symbol['pitch'])
                            custos.set('oct', str(oct))
                            custos.set('pname', pname)
                            if humdrum:
                                
                                humdrum_line = get_humdrum_pitch(oct, pname, line_humdrum)
                            break
            if humdrum:
                humdrum_string.append(humdrum_line)

                if 'custos' in humdrum_line:
                    humdrum_string.append('!!LO:LB:g=z')
                    humdrum_string.append('=-')

        if not saved:                        
            if ix_staff != len(stafflist) - 1:
                barline = etree.SubElement(xml_layer, 'barLine')
                barline.set('{http://www.w3.org/XML/1998/namespace}id',
                            f'barline-{ids[increment_idcounter()]}')
                barline.set('visible', 'false')

    if not saved:
        save_mei_file(metadata, filename_counter, ix_staff, mei, filename, stafflist, humdrum_string, humdrum)
                
        
    cprint('Conversion complete!', 'green')

def get_humdrum_pitch(oct: int, pname: str, line: str) -> str:
    """ converts mei pitch to humdrum pitch """
    if oct == 2:
        line += pname.upper() + pname.upper()
    elif oct == 3:
        line += pname.upper()
    elif oct == 4:
        line += pname
    elif oct == 5:
        line += pname + pname

    return line

def save_mei_file(metadata: list,
                  filename_counter: int,
                  ix_staff: int,
                  mei: etree.Element,
                  filename: str,
                  stafflist: list,
                  humdrum_string: list,
                  humdrum: bool) -> tuple:
    """
    Save the MEI file and optionally a Humdrum file.

    Parameters
    ----------
    metadata : list
        List of metadata for the MEI files.
    filename_counter : int
        Counter for the filename to ensure unique filenames.
    ix_staff : int
        Index of the current staff in the metadata.
    mei : lxml.etree._Element
        The MEI element to be saved.
    filename : str
        The current filename.
    stafflist : list
        List of staffs in the MEI files.
    humdrum_string : list
        List of strings representing the Humdrum content.
    humdrum : bool
        Flag indicating whether to save a Humdrum file.

    Returns
    -------
    tuple
        The updated filename and filename counter.

    """
    global used_filenames

    if filename not in used_filenames:
        save_name = f'{metadata[ix_staff]}_01.mei'
        filename_counter = 1
        used_filenames.append(filename)
    else:
        filename_counter += 1
        save_name = f'{filename}_{str(filename_counter).zfill(2)}.mei'
   

    if ix_staff != len(stafflist) - 1:
        filename = metadata[ix_staff + 1]

    tree = etree.ElementTree(mei)
    xml_content = etree.tostring(tree, pretty_print=True)

    with open(os.path.join("mei_output", save_name), 'wb') as f:
        f.write(xml_content)

    if humdrum:
        with open(os.path.join("humdrum_output", f'{save_name[:-3]}.mens'), 'w') as f:
            f.write('\n'.join(humdrum_string))

    return filename, filename_counter


def create_mei_declarations(ids: list, mei: etree.Element) -> tuple:
    """
    Create MEI declarations and return the clef, key signature,
    and layer elements.

    Parameters
    ----------
    ids : list
        List of IDs to be used for the XML elements.
    mei : lxml.etree.Element
        The root MEI element.

    Returns
    -------
    tuple
        The clef, key signature, and layer XML elements.

    """
    music = etree.SubElement(mei, 'music')
    body = etree.SubElement(music, 'body')
    mdiv = etree.SubElement(body, 'mdiv')
    mdiv.set('{http://www.w3.org/XML/1998/namespace}id',
                 f'mdiv-{ids[increment_idcounter()]}')

    score = etree.SubElement(mdiv, 'score')
    score.set('{http://www.w3.org/XML/1998/namespace}id',
                  f'score-{ids[increment_idcounter()]}')

    scoreDef = etree.SubElement(score, 'scoreDef')
    scoreDef.set('{http://www.w3.org/XML/1998/namespace}id',
                     f'scoreDef-{ids[increment_idcounter()]}')
    staffGrp = etree.SubElement(scoreDef, 'staffGrp')
    staffGrp.set('{http://www.w3.org/XML/1998/namespace}id',
                     f'staffGrp-{ids[increment_idcounter()]}')
    staffDef = etree.SubElement(staffGrp, 'staffDef')
    staffDef.set('{http://www.w3.org/XML/1998/namespace}id',
                     f'staffDef-{ids[increment_idcounter()]}')
    staffDef.set('n', '1')
    staffDef.set('notationtype', 'mensural.white')
    staffDef.set('lines', '5')
    xml_clef = etree.SubElement(staffDef, 'clef')
    xml_keySign = etree.SubElement(staffDef, 'keySig')

    section = etree.SubElement(score, 'section')
    section.set('{http://www.w3.org/XML/1998/namespace}id',
                    f'section-{ids[increment_idcounter()]}')
        
    xml_staff = etree.SubElement(section, 'staff')
    xml_staff.set('{http://www.w3.org/XML/1998/namespace}id',
                        f'staff-{ids[increment_idcounter()]}')
    xml_staff.set('n', '1')

    xml_layer = etree.SubElement(xml_staff, 'layer')
    xml_layer.set('{http://www.w3.org/XML/1998/namespace}id',
                        f'layer-{ids[increment_idcounter()]}')
    xml_layer.set('n', '1')
    return xml_clef, xml_keySign, xml_layer


def create_meihead():
    """
    Create the MEI header element.

    This function creates the MEI header element with the necessary 
    sub-elements and attributes. It sets the application name to 
    'convert_to_mei' and the current date and version number.

    Returns
    -------
    lxml.etree.Element
        The created MEI header element.

    """
    meihead = etree.Element('meiHead')
    fileDesc = etree.SubElement(meihead, 'fileDesc')
    titleStmt = etree.SubElement(fileDesc, 'titleStmt')
    title = etree.SubElement(titleStmt, 'title')
    pubStmt = etree.SubElement(fileDesc, 'pubStmt')

    encodingDesc = etree.SubElement(meihead, 'encodingDesc')
    appInfo = etree.SubElement(encodingDesc, 'appInfo')
    application = etree.SubElement(appInfo, 'application')
    application.set('isodate', datetime.now().isoformat())
    application.set('version', VERSION)
    name = etree.SubElement(application, 'name')
    name.text = 'mensural_to_mei'

    return meihead


def analyse_clef(clef: dict) -> tuple:
    """
    Analyse the clef and return the shape of the clef 
    and the line of the clef. The pitch holds information of
    the detected clef from symbol classification.

    This function takes a dictionary representing a clef and returns 
    the shape and line of the clef based on the pitch.

    Parameters
    ----------
    clef : dict
        Dictionary representing a clef.The 'pitch' key holds
        information of the detected clef.

    Returns
    -------
    tuple
        The shape and line of the clef.

    """
    p = clef['pitch']
    if p == 'c-g':
        shape = 'G'
        line = '2'
    elif p == 'c-c-e':
        shape = 'C'
        line = '1'
    elif p == 'c-c-g':
        shape = 'C'
        line = '2'
    elif p == 'c-c-b':
        shape = 'C'
        line = '3'
    elif p == 'c-c-d':
        shape = 'C'
        line = '4'
    elif p == 'c-f-b':
        shape = 'F'
        line = '3'
    elif p == 'c-f':
        shape = 'F'
        line = '4'

    return shape, line


def analyse_mensuration(symbol: dict) -> tuple:
    """
    Analyse the mensuration and return the mei-definition.

    This function takes a dictionary representing a mensuration and returns 
    the sign and slash of the mensuration based on the pitch.

    Parameters
    ----------
    clef : dict
        Dictionary representing a mensuration. The 'pitch' key holds
        information of the detected mensuration.

    Returns
    -------
    tuple
        The sign and slash of the mensuration.

    """
    p = symbol['pitch']
    if p == 'met_c':
        sign = 'C'
        slash = ''
    elif p == 'al-br':
        sign = 'C'
        slash = '1'
    elif p == 'met_o_cut':
        sign = 'O'
        slash = '1'

    return sign, slash
 

def analyse_note(clef: dict, pitch: str) -> tuple:
    """
    Analyse a note's pitch in relation to a given clef.

    This function takes a dictionary representing a clef and a pitch 
    string. It returns the octave and relative pitch of the note based 
    on the clef and the pitch.

    Parameters
    ----------
    clef : dict
        Dictionary representing a clef. It should have a 'pitch' key.
    pitch : str
        String representing the pitch of the note.

    Returns
    -------
    tuple
        The octave and relative pitch of the note.

    """
    c = clef['pitch']

    # Define a dictionary to map mensural note steps to integers
    MENSURAL_NOTE_STEPS = {'c0': 0, 'd0': 1, 'e0': 2, 'f0': 3, 'g0': 4, 'a0': 5, 'b0': 6,
                           'c1': 7, 'd1': 8, 'e1': 9, 'f1': 10, 'g1': 11, 'a1': 12, 'b1': 13,
                           'c2': 14, 'd2': 15, 'e2': 16, 'f2': 17, 'g2': 18, 'a2': 19, 'b2': 20
                           }
    
    # Define a dictionary to map relative note steps to note names
    RELATIVE_NOTE_STEPS = {
            -14: 'c', -13: 'd', -12: 'e', -11: 'f', -10: 'g', -9: 'a', -8: 'b',
            -7: 'c', -6: 'd', -5: 'e', -4: 'f', -3: 'g', -2: 'a', -1: 'b',
            0: 'c', 1: 'd', 2: 'e', 3: 'f', 4: 'g', 5: 'a', 6: 'b',
            7: 'c', 8: 'd', 9: 'e', 10: 'f', 11: 'g', 12: 'a', 13: 'b'
        }
    
    # Set the initial octave
    octave = 4

    # Determine the clef line based on the 'pitch' of the clef
    if c == 'c-c-g':
        clefline = MENSURAL_NOTE_STEPS['g1']
    elif c == 'c-c-e':
        clefline = MENSURAL_NOTE_STEPS['e1']
    elif c == 'c-c-b':
        clefline = MENSURAL_NOTE_STEPS['b1']
    elif c == 'c-c-d':
        clefline = MENSURAL_NOTE_STEPS['d2']
    elif c == 'c-g':
        clefline = MENSURAL_NOTE_STEPS['c1']
    elif c == 'c-f-b':
        clefline = MENSURAL_NOTE_STEPS['f2']
    elif c == 'c-f':
        clefline = MENSURAL_NOTE_STEPS['a2']

    # Determine the pitch line based on the pitch
    pitchline = MENSURAL_NOTE_STEPS[pitch]

    # Calculate the relative pitch
    rel_pitch = pitchline - clefline

    # Adjust the octave and relative pitch based on the relative pitch
    if -7 <= rel_pitch < 0:
        octave -= 1
        relative_pitch = RELATIVE_NOTE_STEPS[7 + rel_pitch]
    elif rel_pitch > 6:
        octave += 1
        relative_pitch = RELATIVE_NOTE_STEPS[rel_pitch]
    elif -14 <= rel_pitch < -7:
        octave -= 2
        relative_pitch = RELATIVE_NOTE_STEPS[7 + rel_pitch]
    else:
        relative_pitch = RELATIVE_NOTE_STEPS[rel_pitch]

    return octave, relative_pitch

if __name__ == "__main__":
    with open("symbols_and_pitches.pkl", "rb") as f:
        saved_data = pickle.load(f)

    convert_to_mei_and_humdrum(saved_data)