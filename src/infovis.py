import matplotlib.pyplot as plt
import pandas as pd
import pymongo as mongo
import sys

myclient = mongo.MongoClient("mongodb://localhost:27017/")

mydb = myclient["university"]
std_col = mydb["student"]
lec_col = mydb["lecture"]
mod_col = mydb["module"]


def plot_attendance(student_id):
    query = {"_id": f"{student_id}"}
    student = std_col.find(query)
    document_count = sum(1 for _ in student)
    if document_count == 0:
        print(f"No student found for {student_id}.")
        return

    attendance_status = []
    lec_dates = []
    try:
        lectures = lec_col.find({}, {"_id": 0, "date": 1, "attendance": 1})
        for lecture in lectures:
            lec_dates.append(lecture["date"])
            lec_attendance = lecture["attendance"]
            if student_id in lec_attendance:
                attendance_status.append("Present")
            else:
                attendance_status.append("Absent")
    except StopIteration:
        print("No lecture data found.")
        return

    # Figure with two subplots arranged horizontally
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        gridspec_kw={"width_ratios": [1, 1]},
    )

    # Subplot 1: Table for daily attendance status
    lec_attendance_data = {"Date": lec_dates, "Attendance Status": attendance_status}
    df = pd.DataFrame(lec_attendance_data)

    ax1.axis("tight")
    ax1.axis("off")
    table = ax1.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        cellColours=[["lightgray"] * 2] * len(df),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)  # Row and column height

    ax1.set_title(f"Attendance Status for Student {student_id}")

    # Calculate cumulative attendance percentage
    percentage_present = []
    present_count = 0
    total_days_count = 0

    for i in range(len(lec_dates)):
        if attendance_status[i] == "Present":
            present_count += 1
        total_days_count += 1
        percentage_present.append((present_count / total_days_count) * 100)

    # Subplot 2: Line chart for attendance status over time
    ax2.plot(
        lec_dates,
        percentage_present,
        marker="o",
        color="skyblue",
        linestyle="-",
    )
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Attendance Presentage")
    ax2.set_title(f"Attendance Status Over Time")
    ax2.grid(axis="y", linestyle="--", alpha=0.5, which="both")

    # Rotate x-axis labels
    ax2.tick_params(axis="x", rotation=45)

    # Add a gap between the two subplots
    plt.subplots_adjust(wspace=0.4)

    plt.show()


if __name__ == "__main__":
  
    if len(sys.argv) != 2:
        print("Usage: python infovis.py <student_id>")
    else:
        student_id = sys.argv[1]
        plot_attendance(student_id)
