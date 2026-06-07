from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import os


def generate_report(run_folder, accuracy, auc):
    print("\n[Report] Generating PDF report...")

    file_path = os.path.join(run_folder, "report.pdf")

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("Medical AI System Report", styles['Title']))
    content.append(Spacer(1, 20))

    content.append(Paragraph(f"Accuracy: {round(accuracy,4)}", styles['Normal']))
    content.append(Paragraph(f"AUC Score: {round(auc,4)}", styles['Normal']))
    content.append(Spacer(1, 20))

    # Images
    for img in ["roc_curve.png", "confusion_matrix.png", "shap_importance.png"]:
        img_path = os.path.join(run_folder, img)
        if os.path.exists(img_path):
            content.append(Image(img_path, width=400, height=250))
            content.append(Spacer(1, 15))

    doc.build(content)

    print(f"[Report] Saved: {file_path}")