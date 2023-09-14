import cv2
import numpy as np
from PIL import Image, ImageOps
# import subprocess.output
import xml.etree.ElementTree as ET
import pytesseract
import pymongo as mongo
import os

class OcrTool:
    def __init__(self, image_with_lines_detected, perspective_corrected_image, xml_file_path, image_path):
        self.binarized_img = image_with_lines_detected
        self.image = perspective_corrected_image
        self.xml_path = xml_file_path
        self.image_path = f"{image_path}"

    # Store processed image - PIL Image
    def store_process_image(self, output_path, image):
        image.save(output_path)

    def dilate_image(self):
        kernel_to_remove_gaps_between_words = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ])
        img_array = np.array(self.binarized_img)
        self.dilated_image = cv2.dilate(
            img_array, kernel_to_remove_gaps_between_words, iterations=1)
        simple_kernel = np.ones((3, 3), np.uint8)
        self.dilated_image = cv2.dilate(
            self.dilated_image, simple_kernel, iterations=1)
        self.dilated_image = Image.fromarray(self.dilated_image)

    def find_contours(self):
        # Convert PIL Image to numpy array
        dilated_image_array = np.array(self.dilated_image)
        result = cv2.findContours(
            dilated_image_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = result[0]
        # Convert the original image to a numpy array
        original_image_array = np.array(self.image)
        # Create a copy of the original image to draw contours on
        self.image_with_contours_drawn = original_image_array.copy()
        cv2.drawContours(self.image_with_contours_drawn,
                         self.contours, -1, (0, 255, 0), 3)
        self.image_with_contours_drawn = Image.fromarray(
            self.image_with_contours_drawn)

    def approximate_contours(self):
        self.approximated_contours = []
        for contour in self.contours:
            approx = cv2.approxPolyDP(contour, 3, True)
            self.approximated_contours.append(approx)

    def draw_contours(self):
        self.image_with_contours = self.original_image.copy()
        cv2.drawContours(self.image_with_contours,
                         self.approximated_contours, -1, (0, 255, 0), 5)

    def convert_contours_to_bounding_boxes(self):
        self.bounding_boxes = []
        self.image_with_all_bounding_boxes = self.image.copy()
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= 15 and h >= 15:
                self.bounding_boxes.append((x, y, w, h))
            img_array = np.array(self.image_with_all_bounding_boxes)
            self.image_with_all_bounding_boxes = cv2.rectangle(
                img_array, (x, y), (x + w, y + h), (0, 255, 0), 5)
            self.image_with_all_bounding_boxes = Image.fromarray(
                self.image_with_all_bounding_boxes)

    def get_mean_height_of_bounding_boxes(self):
        heights = []
        for bounding_box in self.bounding_boxes:
            x, y, w, h = bounding_box
            heights.append(h)
        return np.mean(heights)

    def sort_bounding_boxes_by_y_coordinate(self):
        self.bounding_boxes = sorted(self.bounding_boxes, key=lambda x: x[1])

    def club_all_bounding_boxes_by_similar_y_coordinates_into_rows(self):
        self.rows = []
        half_of_mean_height = self.mean_height / 2
        current_row = [self.bounding_boxes[0]]
        for bounding_box in self.bounding_boxes[1:]:
            current_bounding_box_y = bounding_box[1]
            previous_bounding_box_y = current_row[-1][1]
            distance_between_bounding_boxes = abs(
                current_bounding_box_y - previous_bounding_box_y)
            if distance_between_bounding_boxes <= half_of_mean_height:
                current_row.append(bounding_box)
            else:
                self.rows.append(current_row)
                current_row = [bounding_box]
        self.rows.append(current_row)

    def sort_all_rows_by_x_coordinate(self):
        for row in self.rows:
            row.sort(key=lambda x: x[0])

    def crop_each_bounding_box_and_ocr(self):
        self.table = []
        current_row = []
        image_number = 0
        original_image_array = np.array(
            self.image)  # Convert to numpy array
        del self.rows[0]
        for row in self.rows:
            for bounding_box in row:
                x, y, w, h = bounding_box
                y = y - 5
                cropped_image = original_image_array[y:y+h, x:x+w]
                image_slice_path = "./uploads/OcrTool/Tesseract/img_" + \
                    str(image_number) + ".jpg"
                cv2.imwrite(image_slice_path, cropped_image)
                gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                results_from_ocr = self.get_result_from_tesseract(gray)
                current_row.append(results_from_ocr)
                image_number += 1
            self.table.append(current_row)
            current_row = []

    def get_result_from_tesseract(self, image_crop):
        # Use pytesseract to extract text from the image
        text = pytesseract.image_to_string(
            image_crop,
            lang='eng',
            config='--oem 3 --psm 7 --dpi 72 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789().calmg* "'
        )
        return text.strip()

    # def generate_csv_file(self):
    #     with open("output.csv", "w") as f:
    #         for row in self.table:
    #             f.write(",".join(row) + "\n")

    def generate_xml_file(self):
        root = ET.Element("nsbm")
        students = ET.SubElement(root, "students")
        batches = ET.SubElement(students, "batches")

        for i, row in enumerate(self.table):
            if len(row) > 1:
                if (row[1].replace(" ", "")).isdigit():
                    student = ET.SubElement(batches, "student")
                    # no = ET.SubElement(student, "no")
                    # no.text = row[0]
                    index = ET.SubElement(student, "index")
                    index.text = row[1]
                    # title = ET.SubElement(student, "title")
                    # title.text = row[2]
                    name = ET.SubElement(student, "name")
                    name.text = row[3]
                    signature = ET.SubElement(student, "attendance")
                    if len(row) < 5:
                        signature.text =  "Absent"
                    else:
                        signature.text = "Present"

        tree = ET.ElementTree(root)
        tree.write(self.xml_path)
    


    def save_results_to_db(self):
        try:
            myclient = mongo.MongoClient("mongodb://localhost:27017/")
            mydb = myclient["university"]
            lec_col = mydb["lecture"]
            std_col = mydb["student"]
            
            date = os.path.splitext(self.image_path)[0]
            present_students = []
            module = "CGV"
            batch = "20.1"
            lecturer = "Dr. Rasika Rananaweera"

            for i, row in enumerate(self.table):
                index = row[1].replace(" ", "")
                if len(row) >= 5 and index.isdigit():
                    present_students.append(row[1])
                query = {"_id": f"{index}"}
                student = std_col.find(query)
                document_count = sum(1 for _ in student)
                if document_count == 0 and index.isdigit():
                    insert_student = std_col.insert_one({"_id":index, "name":row[3], "batch":batch})

            results = {"module": module, "date": date, "batch": batch, "lecturer": lecturer, "attendance": present_students}
            
            insert_result = lec_col.insert_one(results)

            if insert_result.inserted_id:
                print("Attendance saved with ID:", insert_result.inserted_id)
            else:
                print("Insertion failed")
        
        except Exception as e:
            print("Server Error:", e)

        finally:
            myclient.close()


    def execute(self):
        # self.remove_noise_with_erode()
        # self.store_process_image(
        #     './uploads/OcrTool/27_eroded_image.jpg', self.erode_img)
        self.dilate_image()
        self.store_process_image(
           './uploads/OcrTool/28_dilated_image.jpg', self.dilated_image)
        self.find_contours()
        self.store_process_image(
            './uploads/OcrTool/29_contours.jpg', self.image_with_contours_drawn)
        self.convert_contours_to_bounding_boxes()
        self.store_process_image(
            './uploads/OcrTool/30_bounding_boxes.jpg', self.image_with_all_bounding_boxes)
        self.mean_height = self.get_mean_height_of_bounding_boxes()
        self.sort_bounding_boxes_by_y_coordinate()
        self.club_all_bounding_boxes_by_similar_y_coordinates_into_rows()
        self.sort_all_rows_by_x_coordinate()
        self.crop_each_bounding_box_and_ocr()
        # self.generate_csv_file()
        self.generate_xml_file()
        self.save_results_to_db()
