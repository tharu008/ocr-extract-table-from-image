import OcrTool as ot
import TableExtractor as te
import TableLinesDetector as tld
import cv2
import numpy as np


path_to_image = "D:\\University\\YEAR 04 SEM 02\\CGV\\Assignment\\image-processing-model\\src\\uploads\\1.jpeg"
table_extractor = te.TableExtractor(path_to_image)
perspective_corrected_image = table_extractor.execute()


table_lines_remover = tld.TableLinesDetector(perspective_corrected_image)
image_with_lines_detected = table_lines_remover.execute()

ocr_to_table_tool = ot.OcrTool(
    image_with_lines_detected, perspective_corrected_image)
ocr_to_table_tool.execute()

