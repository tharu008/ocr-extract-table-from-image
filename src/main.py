import OcrToTableTool as ottt
import TableExtractor as te
import TableLinesRemover as tlr
import cv2
import numpy as np


path_to_image = "D:\\University\\YEAR 04 SEM 02\\CGV\\Assignment\\image-processing-model\\src\\uploads\\sample2.jpeg"
table_extractor = te.TableExtractor(path_to_image)
perspective_corrected_image = table_extractor.execute()


table_lines_remover = tlr.TableLinesRemover(perspective_corrected_image)
image_without_lines_noise_removed = table_lines_remover.execute()

ocr_to_table_tool = ottt.OcrToTableTool(
    image_without_lines_noise_removed, perspective_corrected_image)
ocr_to_table_tool.execute()
