import OcrTool as ot
import TableExtractor as te
import TableLinesDetector as tld
import cv2
import numpy as np
import sys

if __name__ == "__main__":
  
    if len(sys.argv) != 3:
        print("Usage: python sams.py <xml_file> <image_file>")
    else:
        print("Processing...")
        xml_file = sys.argv[1]
        path_to_image= sys.argv[2]
        table_extractor = te.TableExtractor(path_to_image)
        perspective_corrected_image = table_extractor.execute()

        table_lines_remover = tld.TableLinesDetector(perspective_corrected_image)
        image_with_lines_detected = table_lines_remover.execute()

        ocr_to_table_tool = ot.OcrTool(
            image_with_lines_detected, perspective_corrected_image, xml_file, path_to_image)
        ocr_to_table_tool.execute()

