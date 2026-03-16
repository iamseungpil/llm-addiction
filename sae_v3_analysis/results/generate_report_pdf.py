#!/usr/bin/env python3
"""V3 SAE 분석 보고서 PDF 생성 (reportlab, JupyterLab 호환).
비평 반영 v2: 구조 개선, 사실 오류 수정, 신뢰구간 추가, 방법론 보강."""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
OUTPUT_PDF = os.path.join(RESULTS_DIR, "sae_v3_analysis_report.pdf")

WIDTH, HEIGHT = A4
MARGIN = 2.5 * cm

# ---- Font Registration ----
base = "/usr/share/fonts/truetype/nanum"
pdfmetrics.registerFont(TTFont("NanumGothic", os.path.join(base, "NanumGothic.ttf")))
pdfmetrics.registerFont(TTFont("NanumGothicBold", os.path.join(base, "NanumGothicBold.ttf")))
registerFontFamily("NanumGothic", normal="NanumGothic", bold="NanumGothicBold")
FONT, FONT_BOLD = "NanumGothic", "NanumGothicBold"


def get_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "MainTitle", fontSize=18, spaceAfter=4,
        alignment=TA_CENTER, leading=24, fontName=FONT_BOLD,
    ))
    styles.add(ParagraphStyle(
        "Subtitle", fontSize=11, spaceAfter=12,
        alignment=TA_CENTER, textColor=colors.grey, leading=14, fontName=FONT,
    ))
    styles.add(ParagraphStyle(
        "SectionHead", fontSize=14, spaceBefore=18,
        spaceAfter=8, textColor=colors.HexColor("#1a1a1a"), fontName=FONT_BOLD,
        leading=18,
    ))
    styles.add(ParagraphStyle(
        "SubHead", fontSize=12, spaceBefore=12,
        spaceAfter=6, textColor=colors.HexColor("#333333"), fontName=FONT_BOLD,
        leading=16,
    ))
    styles.add(ParagraphStyle(
        "Body", fontSize=10, leading=15,
        spaceAfter=6, alignment=TA_JUSTIFY, fontName=FONT,
    ))
    styles.add(ParagraphStyle(
        "BulletItem", fontSize=10, leading=15,
        leftIndent=18, spaceAfter=3, bulletIndent=6, alignment=TA_LEFT,
        fontName=FONT,
    ))
    styles.add(ParagraphStyle(
        "Caption", fontSize=9, leading=12,
        spaceAfter=10, alignment=TA_CENTER, textColor=colors.HexColor("#555555"),
        fontName=FONT,
    ))
    styles.add(ParagraphStyle(
        "SmallNote", fontSize=8, leading=10,
        spaceAfter=4, textColor=colors.grey, fontName=FONT,
    ))
    styles.add(ParagraphStyle(
        "Transition", fontSize=10, leading=15,
        spaceAfter=6, spaceBefore=10, alignment=TA_JUSTIFY, fontName=FONT,
        textColor=colors.HexColor("#444444"),
    ))
    return styles


def make_table(data, col_widths=None, header=True):
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 13),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 1.2, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1.2, colors.black),
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, colors.black),
    ]
    if header:
        style_cmds += [
            ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
        ]
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    t.setStyle(TableStyle(style_cmds))
    return t


def add_figure(story, filename, caption_text, styles, width=None):
    path = os.path.join(FIGURES_DIR, filename)
    if not os.path.exists(path):
        story.append(Paragraph(f"[Figure missing: {filename}]", styles["Body"]))
        return
    if width is None:
        width = WIDTH - 2 * MARGIN
    img = Image(path, width=width, height=width * 0.55)
    img.hAlign = "CENTER"
    story.append(img)
    story.append(Paragraph(caption_text, styles["Caption"]))


def B(text):
    return f'<b>{text}</b>'


def I(text):
    return f'<i>{text}</i>'


def build_report():
    doc = SimpleDocTemplate(
        OUTPUT_PDF, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="V3 SAE/Activation Analysis Report",
        author="SAE Analysis Pipeline v3",
    )

    styles = get_styles()
    story = []

    # ================================================================
    # 표지
    # ================================================================
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("V3 SAE/Activation 분석 보고서", styles["MainTitle"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Gemma-2-9B의 도박 유사 행동에 대한 신경 서명 분석<br/>"
        "세 가지 도박 패러다임 간 비교",
        styles["Subtitle"]
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "SAE Analysis Pipeline v3 | LLM 도박 중독 프로젝트 (NMI 투고) | 2026년 3월 6일",
        styles["SmallNote"]
    ))
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 10))

    # ================================================================
    # 1. 핵심 요약
    # ================================================================
    story.append(Paragraph("1. 핵심 요약", styles["SectionHead"]))

    story.append(Paragraph(
        "본 분석은 네 가지 핵심 질문에 답하기 위해 설계되었다: "
        "(A) 내부 표현으로 파산 게임을 분류할 수 있는가? "
        "(B) 게임 초반부터 파산을 예측할 수 있는가? "
        "(C) 한 패러다임의 분류기가 다른 패러다임에서도 작동하는가? "
        "(D) SAE 피처가 원시 hidden state 대비 이점이 있는가?",
        styles["Body"]
    ))

    exec_data = [
        ["목표", "과제", "IC", "SM", "MW", "핵심 발견"],
        ["A (분류)", "SAE 최고 AUC", "0.964\u00b10.004", "0.981\u00b10.005", "0.966\u00b10.008", "모두 >0.96"],
        ["A (분류)", "Hidden 최고 AUC", "0.963\u00b10.005", "0.982\u00b10.005", "0.968\u00b10.007", "SAE와 동등"],
        ["B (조기예측)", "R1 AUC", "0.852", "0.895", "0.766", "1라운드부터 탐지"],
        ["C (전이)", "교차도메인", "0.812-0.916", "", "", "평균 유지율 88.6%"],
        ["D (비교)", "SAE vs Hidden", "\u0394AUC < 0.005", "", "", "실질적 차이 없음"],
    ]
    story.append(make_table(exec_data, col_widths=[52, 78, 78, 78, 78, 100]))
    story.append(Paragraph(
        "Table 1: 전체 분석 목표별 핵심 결과. 5-fold 층화 CV, balanced class weight 적용. "
        "\u00b1 값은 fold간 표준편차.",
        styles["Caption"]
    ))

    story.append(Paragraph(
        f"{B('결론:')} 세 가지 도박 패러다임 모두에서 파산 게임과 안전 게임을 "
        "AUC > 0.96으로 구분할 수 있다. SAE 피처와 원시 hidden state 모두 동등한 성능을 보이며, "
        "이 신호는 게임 첫 라운드부터 존재한다 (R1 AUC 0.77-0.90). "
        "교차 도메인 전이 AUC 0.81-0.92는 과제 프레이밍이 달라도 공유된 신경 메커니즘이 "
        "도박 유사 행동의 기저에 있음을 나타낸다.",
        styles["Body"]
    ))

    # ================================================================
    # 2. 실험 설계
    # ================================================================
    story.append(Paragraph("2. 실험 설계", styles["SectionHead"]))

    story.append(Paragraph("2.1 데이터", styles["SubHead"]))
    story.append(Paragraph(
        "세 가지 도박 패러다임의 클린 데이터(V2role/V4role, ROLE_INSTRUCTION 적용, "
        "파서 오류 0%)를 사용하였다.",
        styles["Body"]
    ))

    data_tbl = [
        ["패러다임", "버전", "게임 수", "라운드", "파산(BK)", "BK율"],
        ["Investment Choice (IC)", "V2role", "1,600", "9,119", "172", "10.8%"],
        ["Slot Machine (SM)", "V4role", "3,200", "21,421", "87", "2.7%"],
        ["Mystery Wheel (MW)", "V2role", "3,200", "15,749", "54", "1.7%"],
    ]
    story.append(make_table(data_tbl, col_widths=[115, 48, 52, 52, 52, 42]))
    story.append(Paragraph(
        "Table 2: 데이터셋 요약. BK = 잔고 $0 도달. 모두 Gemma-2-9B-IT 사용.",
        styles["Caption"]
    ))

    story.append(Paragraph("2.2 피처 추출", styles["SubHead"]))
    story.append(Paragraph(
        f"{B('Hidden State:')} Gemma-2-9B-IT의 42개 레이어에서 residual stream 활성화를 추출. "
        "각 라운드마다 마지막 토큰 위치에서 3,584차원 벡터를 얻는다.",
        styles["Body"]
    ))
    story.append(Paragraph(
        f"{B('SAE 피처:')} GemmaScope SAE 인코딩(레이어당 131,072개 피처). "
        "Hidden state를 희소 활성화 벡터로 분해한다(>99.8% 희소성). "
        "레이어에 따라 활성화율 \u22651%인 피처가 80-1,700개 남는다.",
        styles["Body"]
    ))
    story.append(Paragraph(
        f"{B('집계 방식:')} Decision-point 추출(각 게임의 마지막 라운드) \u2014 "
        "게임 종료 직전 모델의 최종 내부 상태를 사용.",
        styles["Body"]
    ))

    story.append(Paragraph("2.3 분류 방법", styles["SubHead"]))
    for item in [
        f"{B('분류기:')} Logistic Regression (L2, C=1.0, lbfgs, max_iter=1000)",
        f"{B('클래스 균형:')} class_weight='balanced' (역빈도 가중치)",
        f"{B('검증:')} 5-fold 층화 교차검증",
        f"{B('지표:')} AUC-ROC (임계값 독립적, 클래스 불균형에 강건)",
        f"{B('피처 필터링:')} 활성화율 < 1%인 SAE 피처 제거",
        f"{B('전처리:')} StandardScaler (fold별 train에서 fit, test에 transform)",
    ]:
        story.append(Paragraph(item, styles["BulletItem"], bulletText="\u2022"))

    story.append(Paragraph("2.4 교차 도메인 전이 방법", styles["SubHead"]))
    story.append(Paragraph(
        "소스 패러다임에서 학습한 분류기를 타겟 패러다임에서 테스트한다. "
        "두 패러다임에서 모두 활성화율 \u22651%인 피처의 교집합(shared features)만 사용하며, "
        "레이어별로 공유 피처 수가 다르다(154-1,085개). "
        "StandardScaler는 소스에서 fit하여 타겟에 적용한다. "
        "최적 레이어는 타겟의 전이 AUC가 최대인 레이어로 선택한다.",
        styles["Body"]
    ))

    # ================================================================
    # 3. Goal A: 파산 분류
    # ================================================================
    story.append(PageBreak())
    story.append(Paragraph("3. Goal A: 파산 분류", styles["SectionHead"]))
    story.append(Paragraph(
        I("모델의 내부 표현으로 파산 게임과 안전 게임을 구분할 수 있는가?"),
        styles["Body"]
    ))

    story.append(Paragraph(
        "세 패러다임 모두에서 decision-point 피처로 AUC > 0.96의 파산 예측이 가능하다. "
        "SAE 피처와 hidden state 모두 유사한 성능을 보인다.",
        styles["Body"]
    ))

    add_figure(story, "report_goal_a_sae_vs_hidden.png",
               "Figure 1: 레이어별 BK 분류 AUC. SAE 피처(색상)와 원시 hidden state(회색) 비교. "
               "별표는 표시된 레이어 중 최고치. 실제 전체 최적 레이어는 본문 참조. "
               "두 표현 모두 유사한 피크 AUC를 달성하나, 레이어 프로파일이 다르다.",
               styles)

    for item in [
        f"{B('IC:')} SAE L22 AUC=0.964\u00b10.004, Hidden L25 AUC=0.963\u00b10.005. "
        "중간 레이어에서 역U자 프로파일.",
        f"{B('SM:')} SAE L12 AUC=0.981\u00b10.005, Hidden L10 AUC=0.982\u00b10.005. "
        "초기 레이어에서 위험 신호를 강하게 인코딩.",
        f"{B('MW:')} SAE L33 AUC=0.966\u00b10.008, Hidden L20 AUC=0.968\u00b10.007. "
        "적은 BK 수(n=54)로 인해 레이어간 분산이 높음.",
    ]:
        story.append(Paragraph(item, styles["BulletItem"], bulletText="\u2022"))

    story.append(Paragraph(
        f"{B('레이어 프로파일의 함의:')} SM은 초기 레이어(L10-18)에서, "
        "IC는 중간 레이어(L22-25)에서, MW는 후기 레이어(L33)에서 위험 정보를 인코딩한다. "
        "이는 패러다임별 정보 처리 단계가 다름을 나타낸다.",
        styles["Body"]
    ))

    add_figure(story, "report_goal_a_f1_comparison.png",
               "Figure 2: 최고 AUC 레이어에서의 F1 점수. IC가 F1 0.72-0.74로 가장 높고, "
               "SM 0.49-0.53, MW 0.25-0.27 순. 클래스 불균형의 영향을 반영한다.",
               styles, width=(WIDTH - 2 * MARGIN) * 0.75)

    story.append(Paragraph("3.1 SAE vs Hidden State 성능 비교 (Goal D)", styles["SubHead"]))
    add_figure(story, "report_goal_d_feature_vs_hidden.png",
               "Figure 3: 최고 레이어 AUC 비교. SAE 피처(131K sparse) vs hidden state(3,584 dense). "
               "차이는 무시할 수준(\u0394AUC < 0.005).",
               styles, width=(WIDTH - 2 * MARGIN) * 0.75)

    story.append(Paragraph(
        f"SAE 피처와 원시 hidden state는 {B('실질적으로 동등한')} "
        "분류 성능을 보인다(\u0394AUC < 0.005). "
        "SAE의 진정한 이점은 예측 성능이 아닌 {B('해석가능성')}이다: "
        "131K개의 명명된 피처로 어떤 개념이 분류를 구동하는지 식별 가능하나, "
        "3,584개의 hidden 차원은 불투명하다. "
        "또한 SAE 피처는 ~37배 더 희소(활성 80-1,362개 vs dense 3,584개)하여 "
        "효율적 저장 및 피처 수준 분석이 가능하다.",
        styles["Body"]
    ))

    story.append(Paragraph("3.2 클래스 불균형 검증", styles["SubHead"]))
    story.append(Paragraph(
        "높은 AUC가 클래스 불균형의 아티팩트가 아님을 다음과 같이 검증하였다:",
        styles["Body"]
    ))
    for item in [
        "class_weight='balanced'로 소수 클래스의 기여를 비례적으로 보장.",
        "AUC-ROC는 임계값 독립적이며 모든 operating point에서 순위 품질 평가.",
        "IC 순열 검정(N=1000): 귀무분포 최대 AUC = 0.588, 관측 AUC = 0.964, p < 0.001.",
        "Hidden state(3,584 dense)와 SAE 피처(80-1,362 active sparse)가 동등한 AUC \u2192 "
        "차원 수에 의한 아티팩트 배제.",
        f"{B('주의:')} MW(fold당 BK \u224811건)에서는 소표본으로 fold간 분산이 "
        "상대적으로 크다(std=0.008 vs IC의 0.004).",
    ]:
        story.append(Paragraph(item, styles["BulletItem"], bulletText="\u2022"))

    # ================================================================
    # 4. Goal B: 조기 예측
    # ================================================================
    story.append(Paragraph(
        "Goal A에서 게임 종료 시점의 분류가 가능함을 확인하였다. "
        "자연스러운 후속 질문은 이 신호가 게임 초반에도 존재하는지 여부이다.",
        styles["Transition"]
    ))
    story.append(Paragraph("4. Goal B: 조기 예측", styles["SectionHead"]))
    story.append(Paragraph(
        I("게임 초반부터 최종 파산 여부를 예측할 수 있는가?"),
        styles["Body"]
    ))

    add_figure(story, "report_goal_b_early_prediction.png",
               "Figure 4: L22에서의 라운드별 BK 예측 AUC. 세 패러다임 모두 "
               "1라운드부터 chance(0.5) 이상의 예측력을 보인다. "
               "일부 라운드만 표시; 전체 수치는 본문 참조.",
               styles, width=(WIDTH - 2 * MARGIN) * 0.85)

    for item in [
        f"{B('R1 예측이 매우 강력:')} IC AUC=0.852, SM AUC=0.895, MW AUC=0.766. "
        "첫 번째 의사결정 시점의 내부 상태에 이미 최종 게임 결과 정보가 포함되어 있다.",
        f"{B('IC는 R2 이후 감소:')} R2에서 AUC=0.873으로 정점 후 BK 게임이 종료되면서 감소 "
        "(R10 시점에 BK 8건만 남음, AUC 0.457).",
        f"{B('SM은 R5까지 AUC>0.80 유지:')} R6-R8에서 0.73-0.80으로 하락한 뒤 "
        "R9-R10에서 0.81-0.82로 회복. 후반(R25+)의 높은 AUC는 소표본 효과.",
        f"{B('MW는 R3에서 정점:')} AUC=0.851(R3), R5 이후 BK 18건만 남아 소표본 노이즈 증가.",
    ]:
        story.append(Paragraph(item, styles["BulletItem"], bulletText="\u2022"))

    story.append(Paragraph(
        f"{B('핵심 통찰:')} R1의 강한 신호는 모델의 '위험 태도'가 게임 시작 시점부터 "
        "인코딩됨을 의미한다. 이는 게임 결과 피드백이 아닌, 초기 프롬프트 처리가 "
        "위험 궤적을 설정한다는 가설을 지지한다.",
        styles["Body"]
    ))

    early_data = [
        ["패러다임", "R1 AUC", "R1 게임 수", "R1 BK", "레이어"],
        ["Investment Choice", "0.852", "1,600", "172", "L22"],
        ["Slot Machine", "0.895", "3,200", "87", "L22"],
        ["Mystery Wheel", "0.766", "3,200", "54", "L22"],
    ]
    story.append(make_table(early_data, col_widths=[100, 60, 65, 50, 50]))
    story.append(Paragraph(
        "Table 3: 1라운드 조기 예측 상세. 모든 패러다임에서 L22 사용.",
        styles["Caption"]
    ))

    # ================================================================
    # 5. Goal C: 교차 도메인 전이
    # ================================================================
    story.append(PageBreak())
    story.append(Paragraph(
        "Goal A-B에서 각 패러다임 내 분류 및 조기 예측이 가능함을 확인하였다. "
        "다음 질문은 이러한 신경 서명이 패러다임 간에 공유되는지 여부이다.",
        styles["Transition"]
    ))
    story.append(Paragraph("5. Goal C: 교차 도메인 전이", styles["SectionHead"]))
    story.append(Paragraph(
        I("공유된 신경 피처가 서로 다른 도박 과제에서도 파산을 예측하는가?"),
        styles["Body"]
    ))

    add_figure(story, "report_goal_c_transfer_matrix.png",
               "Figure 5: 교차 도메인 전이 AUC. 대각선: within-domain(최고 SAE AUC). "
               "비대각선: 소스 패러다임에서 학습, 타겟에서 테스트.",
               styles, width=(WIDTH - 2 * MARGIN) * 0.65)

    transfer_data = [
        ["전이 방향", "최적 레이어", "공유 피처 수", "전이 AUC", "타겟 Within", "유지율"],
        ["IC \u2192 MW", "L40", "535", "0.916", "0.966", "94.8%"],
        ["IC \u2192 SM", "L16", "154", "0.894", "0.981", "91.1%"],
        ["MW \u2192 SM", "L36", "656", "0.882", "0.981", "89.9%"],
        ["MW \u2192 IC", "L38", "1,085", "0.833", "0.964", "86.4%"],
        ["SM \u2192 IC", "L40", "459", "0.820", "0.964", "85.1%"],
        ["SM \u2192 MW", "L16", "160", "0.812", "0.966", "84.1%"],
        ["평균", "", "", "0.859", "", "88.6%"],
    ]
    story.append(make_table(transfer_data, col_widths=[62, 52, 58, 58, 62, 48]))
    story.append(Paragraph(
        "Table 4: 6개 방향별 교차 도메인 전이 AUC. '공유 피처 수'는 두 패러다임에서 "
        "모두 활성화율 \u22651%인 SAE 피처의 교집합 크기.",
        styles["Caption"]
    ))

    for item in [
        f"{B('6개 쌍 모두 AUC 0.80 초과:')} 범위 0.812-0.916, 평균 0.859. "
        "패러다임 간 공유 신경 메커니즘의 강력한 증거.",
        f"{B('IC가 소스로서 가장 우수:')} IC\u2192MW(0.916), IC\u2192SM(0.894). "
        "IC의 BK 수(n=172)가 가장 많아 더 나은 학습 신호를 제공.",
        f"{B('비대칭 전이:')} IC\u2192SM(0.894) > SM\u2192IC(0.820). "
        "IC가 더 일반화 가능한 피처를 포착함을 암시.",
        f"{B('평균 유지율 88.6%:')} 한 패러다임에서 학습한 분류기가 다른 패러다임에서 "
        "within-domain AUC의 88.6%를 유지. 도박 위험의 신경 표현이 과제 프레이밍을 "
        "넘어 공유됨을 입증.",
    ]:
        story.append(Paragraph(item, styles["BulletItem"], bulletText="\u2022"))

    # ================================================================
    # 6. 한계점
    # ================================================================
    story.append(Paragraph("6. 한계점", styles["SectionHead"]))
    for i, item in enumerate([
        f"{B('클래스 불균형:')} SM(87/3200=2.7%), MW(54/3200=1.7%)의 적은 양성 사례로 "
        "F1 신뢰성 제한 및 fold간 분산 증가. MW는 fold당 BK \u224811건으로 "
        "소표본 불안정성이 높다.",
        f"{B('SM/MW 순열 검정 미실시:')} IC에서만 p<0.001 확인. SM/MW에서도 "
        "동일 수준의 통계적 유의성이 기대되나, 공식 검증이 필요하다.",
        f"{B('단일 모델:')} 모든 결과가 Gemma-2-9B-IT에서 도출. "
        "LLaMA-3.1-8B 교차 모델 검증이 필요하다.",
        f"{B('Decision-point만 사용:')} 현재 마지막 라운드 피처만 사용. "
        "Game-mean/max 집계(IC V2에서 AUC\u22480.947-0.949)는 SM/MW에 미적용.",
        f"{B('전이 레이어 선택:')} 최적 전이 레이어가 쌍마다 다름(L16-L40). "
        "서로 다른 깊이에서 다른 피처가 일반화됨을 함의하나, "
        "이 레이어 선택이 타겟 데이터에 의존(oracle selection)하므로 과적합 가능성이 있다.",
        f"{B('선형 분류기만 사용:')} Logistic Regression이 비선형 패턴을 과소적합할 수 있으나, "
        "높은 성능은 BK 신호가 대체로 선형 분리 가능함을 함의한다.",
    ], 1):
        story.append(Paragraph(f"{i}. {item}", styles["BulletItem"]))

    # ================================================================
    # 7. 논문에 대한 시사점
    # ================================================================
    story.append(Paragraph("7. 논문에 대한 시사점", styles["SectionHead"]))

    story.append(Paragraph(
        "위 한계점을 감안하더라도, 본 분석 결과는 논문의 핵심 주장을 강화한다. "
        "구체적인 반영 방안은 다음과 같다:",
        styles["Body"]
    ))

    story.append(Paragraph("7.1 Section 3.2 (신경 메커니즘) 반영 사항", styles["SubHead"]))
    for item in [
        f"{B('추가할 그림:')} Figure 5(전이 행렬)를 Section 3.2에 삽입하여 "
        "교차 도메인 일반화를 시각적으로 제시.",
        f"{B('수정할 주장:')} 기존 단일 패러다임(IC) AUC 0.96 → "
        "'세 가지 독립 패러다임에서 AUC 0.96-0.98'로 확장.",
        f"{B('추가할 근거:')} R1 조기 예측(AUC 0.77-0.90)은 "
        "'프롬프트 처리 단계에서 위험 궤적이 결정됨'이라는 논문의 가설을 직접 지지.",
    ]:
        story.append(Paragraph(item, styles["BulletItem"], bulletText="\u2022"))

    story.append(Paragraph("7.2 Section 4 (토론) 반영 사항", styles["SubHead"]))
    for item in [
        f"{B('SAE \u2248 Hidden:')} 동등한 예측 성능은 SAE 피처를 해석가능성 분석에 "
        "사용하는 것이 정보 손실 없이 정당화됨을 논의에 명시.",
        f"{B('\"도박 성향\" 내부 상태:')} 교차 도메인 전이(유지율 88.6%)는 "
        "창발적 미스정렬(emergent misalignment) 논의를 지지 \u2014 "
        "과제별 패턴 매칭이 아닌 일반화 가능한 위험 추구 표현의 형성.",
        f"{B('F1 보고:')} SM/MW의 낮은 F1(0.27-0.53)은 낮은 기저율의 결과이며, "
        "AUC와 함께 보고하여 해석의 맥락을 제공해야 함.",
    ]:
        story.append(Paragraph(item, styles["BulletItem"], bulletText="\u2022"))

    # ================================================================
    # 8. 후속 분석 계획
    # ================================================================
    story.append(Paragraph("8. 후속 분석 계획", styles["SectionHead"]))

    story.append(Paragraph(
        "아래 과제를 영향도(Impact)와 소요시간(Cost) 기준으로 정리하였다:",
        styles["Body"]
    ))

    plan_data = [
        ["우선순위", "과제", "소요시간", "영향도", "설명"],
        ["1", "순열 검정 (SM/MW)", "~30분", "높음",
         "통계적 유의성 공식 검증. 논문 필수."],
        ["2", "핵심 피처 식별", "~1시간", "높음",
         "LR 계수 상위 100개 SAE 피처 \u2192 GemmaScope 라벨 교차참조."],
        ["3", "다층 앙상블", "~2시간", "중간",
         "여러 레이어 결합으로 R1 조기예측 AUC 개선 가능성 검증."],
        ["4", "활성화 패칭", "~4시간(GPU)", "높음",
         "상위 피처의 인과적 역할 검증. 논문 Section 3.2 핵심."],
        ["5", "PCA 궤적 시각화", "~2시간", "중간",
         "BK vs 안전 게임의 라운드별 hidden state 궤적."],
        ["6", "LLaMA 복제", "~8시간(GPU)", "중간",
         "교차 모델 일반성 확인. 투고 시점에 따라 선택."],
    ]
    story.append(make_table(plan_data, col_widths=[42, 88, 50, 42, 160]))
    story.append(Paragraph(
        "Table 5: 후속 분석 계획. 우선순위 1-2는 즉시 실행 가능(GPU 불필요).",
        styles["Caption"]
    ))

    # ================================================================
    # 부록
    # ================================================================
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Paragraph("부록: 실험 파라미터", styles["SectionHead"]))

    param_data = [
        ["파라미터", "값"],
        ["모델", "Gemma-2-9B-IT (google/gemma-2-9b-it)"],
        ["SAE", "GemmaScope 131K (google/gemma-scope-9b-pt-res)"],
        ["레이어", "0-41 (42개)"],
        ["Hidden 차원", "3,584"],
        ["SAE 피처 수", "레이어당 131,072개"],
        ["분류기", "Logistic Regression (L2, C=1.0, balanced)"],
        ["교차검증", "5-fold 층화"],
        ["피처 필터", "활성화율 \u2265 1%"],
        ["랜덤 시드", "42"],
        ["하드웨어", "2x NVIDIA A100-SXM4-40GB"],
        ["추출 시간", "IC: 59분, SM: 220분, MW: 148분"],
        ["분석 시간", "~50분 (Goal A-C)"],
    ]
    story.append(make_table(param_data, col_widths=[100, 300]))

    doc.build(story)
    print(f"PDF 생성 완료: {OUTPUT_PDF} ({os.path.getsize(OUTPUT_PDF):,} bytes)")


if __name__ == "__main__":
    build_report()
