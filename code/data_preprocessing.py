import os
import pandas as pd
from tqdm import tqdm

import xml.etree.ElementTree as ET

import paths

# xml_dir = '/home/ubuntu/nlp_project/Data/ecgen-radiology'
#
# img_dir = '/home/ubuntu/nlp_project/Data/images'

# def create_img_report_df():
#     cols = ['image_id', 'image_path', 'findings', 'impression']
#     data = []
#
#     for file in tqdm(os.listdir(paths.reports_dir_path)):
#         print('file names')
#         print(file)
#         file_path = os.path.join(paths.reports_dir_path, file)
#         if file_path.endswith('.xml'):
#             xml_tree = ET.parse(file_path)
#             findings = xml_tree.find(".//AbstractText[@Label='FINDINGS']").text
#             impression = xml_tree.find(".//AbstractText[@Label='IMPRESSION']").text
#
#             for parent_img in xml_tree.findall('parentImage'):
#                 parent_img_id = parent_img.attrib['id'] + '.png'
#                 img_path = os.path.join(paths.images_dir_path, parent_img_id)
#                 data.append([parent_img_id, img_path, findings, impression])
#
#     image_report_df = pd.DataFrame(data, columns=cols)
#     return image_report_df
#
#
#
#
#
# df = create_img_report_df()
# print(df.head())
# print(df.shape)
# print(df.columns)


def extract_findings_and_impression(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Extract Findings
    findings_start = content.find("FINDINGS:")
    impression_start = content.find("IMPRESSION:")

    findings = ""
    impression = ""

    if findings_start != -1:
        findings = content[findings_start + len("FINDINGS:"):impression_start].strip()

    if impression_start != -1:
        impression = content[impression_start + len("IMPRESSION:"):].strip()

    return findings, impression


def contains_only_dicom(directory_path):
    files = os.listdir(directory_path)
    dicom_files = [f for f in files if f.endswith('.dcm')]
    return len(dicom_files) == len(files) and len(dicom_files) > 0


# reports_root_path = r'/home/ubuntu/nlp_project/Code/physionet.org/files/mimic-cxr/2.1.0/files'


reports_root_path = input("Enter the root path for reports: ").strip()

# Ensure the path exists
if not os.path.exists(reports_root_path):
    raise FileNotFoundError(f"The specified path does not exist: {reports_root_path}")


grp_folders = os.listdir(reports_root_path)

# for p_grp in grp_folders:
#     cxr_path = os.path.join(reports_root_path, p_grp)
#     # print(cxr_path)
#     p_files = os.listdir(cxr_path)
#     # print(p_files)
#     for p in p_files:
#         res_path = os.path.join(cxr_path, p)
#
#
#
#
#         print(res_path)
#
#         if os.path.isdir(res_path):
#             txt_files = [f for f in os.listdir(res_path) if f.endswith('.txt') and f.startswith('s')]
#
#             for s in txt_files:
#                 file_path = os.path.join(res_path, s)
#
#                 print(file_path)
#
#                 findings, impression = extract_findings_and_impression(file_path)
#
#                 print(f"File: {s}")
#                 print(f"Findings:\n{findings}")
#                 print(f"Impression:\n{impression}")
#                 print("=" * 80)

            # for txt_file in txt_files:
            #     txt_file_path = os.path.join(res_path, txt_file)
            #     print(txt_file_path)

    #     if os.path.isdir(res_path):
    #         txt_files = [f for f in os.listdir(res_path) if f.endswith('.txt')]
    #         print(txt_files)


data = []

for p_grp in grp_folders:
    cxr_path = os.path.join(reports_root_path, p_grp)
    p_files = os.listdir(cxr_path)

    for p in p_files:
        res_path = os.path.join(cxr_path, p)

        # print(os.listdir(res_path))
        # Check if the path is a directory
        if os.path.isdir(res_path):
            dicom_dirs = [d for d in os.listdir(res_path) if os.path.isdir(os.path.join(res_path, d))]
            txt_files = [f for f in os.listdir(res_path) if f.endswith('.txt') and f.startswith('s')]

            # print(dicom_dirs)
            # print(txt_files)

            for dicom_dir in dicom_dirs:
                dicom_path = os.path.join(res_path, dicom_dir)
                # print(os.listdir(dicom_path))
                dicom_files = [os.path.join(dicom_path, f) for f in os.listdir(dicom_path) if f.endswith('.dcm')]
                print(dicom_files)

                report_file = f"{dicom_dir}.txt"
                if report_file in txt_files:
                    report_path = os.path.join(res_path, report_file)
                    findings, impressions = extract_findings_and_impression(report_path)

                    for dicom_file in dicom_files:
                        dicom_id = os.path.basename(dicom_file)
                        data.append({
                            "dicom_path": dicom_file,
                            "dicom_id": dicom_id,
                            "findings": findings,
                            "impressions": impressions
                        })

            # if contains_only_dicom(res_path):
            #     dicom_files = [os.path.join(res_path, f) for f in os.listdir(res_path) if f.endswith('.dcm')]
            #     print(dicom_files)
                # Find corresponding text files
                # txt_files = [f for f in os.listdir(res_path) if f.endswith('.txt') and f.startswith('s')]

                # for txt_file in txt_files:
                #     txt_file_path = os.path.join(res_path, txt_file)
                #     findings, impressions = extract_findings_and_impression(txt_file_path)
                #
                #     for dicom_file in dicom_files:
                #         dicom_id = os.path.basename(dicom_file)
                #         data.append({
                #             "dicom_path": dicom_file,
                #             "dicom_id": dicom_id,
                #             "findings": findings,
                #             "impressions": impressions
                #         })

df = pd.DataFrame(data)

print(df.head())

print(len(df))


# df.to_csv('data_crazy_new.csv', index=False)

