import os
import pandas as pd
from tqdm import tqdm

import xml.etree.ElementTree as ET

import paths

# xml_dir = '/home/ubuntu/nlp_project/Data/ecgen-radiology'
#
# img_dir = '/home/ubuntu/nlp_project/Data/images'

def create_img_report_df():
    cols = ['image_id', 'image_path', 'findings', 'impression']
    data = []

    for file in tqdm(os.listdir(paths.reports_dir_path)):
        file_path = os.path.join(paths.reports_dir_path, file)
        if file_path.endswith('.xml'):
            xml_tree = ET.parse(file_path)
            findings = xml_tree.find(".//AbstractText[@Label='FINDINGS']").text
            impression = xml_tree.find(".//AbstractText[@Label='IMPRESSION']").text

            for parent_img in xml_tree.findall('parentImage'):
                parent_img_id = parent_img.attrib['id'] + '.png'
                img_path = os.path.join(paths.images_dir_path, parent_img_id)
                data.append([parent_img_id, img_path, findings, impression])

    image_report_df = pd.DataFrame(data, columns=cols)
    return image_report_df





df = create_img_report_df()
print(df.head())
print(df.shape)
print(df.columns)


