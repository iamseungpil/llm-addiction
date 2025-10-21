#!/usr/bin/env python3
"""
3_1과 3_2 논문의 table citation과 label 일치성 검사
"""

import re
from pathlib import Path

def extract_citations_and_labels(file_path):
    """파일에서 table citation과 label 추출"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # citation 찾기 (\ref{tab:xxx})
    citations = re.findall(r'\\ref\{(tab:[^}]+)\}', content)
    
    # label 찾기 (\label{tab:xxx})
    labels = re.findall(r'\\label\{(tab:[^}]+)\}', content)
    
    return citations, labels

def check_paper_consistency(paper_path, paper_name):
    """논문의 citation-label 일치성 확인"""
    print(f"🔍 {paper_name} 검사")
    print("=" * 50)
    
    citations, labels = extract_citations_and_labels(paper_path)
    
    print(f"Citations 발견: {len(citations)}개")
    for cite in sorted(set(citations)):
        print(f"  \\ref{{{cite}}}")
    
    print(f"\nLabels 발견: {len(labels)}개")
    for label in sorted(set(labels)):
        print(f"  \\label{{{label}}}")
    
    # 누락된 label 찾기
    missing_labels = set(citations) - set(labels)
    if missing_labels:
        print(f"\n❌ 누락된 labels:")
        for missing in sorted(missing_labels):
            print(f"  {missing}")
    
    # 사용되지 않는 label 찾기
    unused_labels = set(labels) - set(citations)
    if unused_labels:
        print(f"\n⚠️  사용되지 않는 labels:")
        for unused in sorted(unused_labels):
            print(f"  {unused}")
    
    # 일치 여부
    if not missing_labels and not unused_labels:
        print(f"\n✅ 모든 citation-label이 일치함")
    else:
        print(f"\n⚠️  불일치 발견")
    
    return citations, labels, missing_labels, unused_labels

def main():
    print("📋 Table Citation-Label 일치성 검사")
    print("=" * 70)
    
    papers = [
        ("/home/ubuntu/llm_addiction/writing/3_1_can_llm_be_addicted_fixed.tex", "3_1 GPT 논문"),
        ("/home/ubuntu/llm_addiction/writing/3_2_llama_feature_analysis.tex", "3_2 LLaMA 논문")
    ]
    
    all_issues = []
    
    for paper_path, paper_name in papers:
        citations, labels, missing, unused = check_paper_consistency(paper_path, paper_name)
        
        if missing or unused:
            all_issues.append({
                'paper': paper_name,
                'path': paper_path,
                'missing': missing,
                'unused': unused
            })
        
        print("\n" + "-" * 70 + "\n")
    
    # 수정이 필요한 사항 요약
    if all_issues:
        print("🔧 수정이 필요한 사항들:")
        print("=" * 50)
        
        for issue in all_issues:
            print(f"\n{issue['paper']}:")
            if issue['missing']:
                print(f"  ❌ 누락된 labels: {len(issue['missing'])}개")
                for missing in sorted(issue['missing']):
                    print(f"    - {missing}")
            if issue['unused']:
                print(f"  ⚠️  미사용 labels: {len(issue['unused'])}개")
                for unused in sorted(issue['unused']):
                    print(f"    - {unused}")
    else:
        print("✅ 모든 논문의 citation-label이 정확히 일치합니다!")
    
    return all_issues

if __name__ == "__main__":
    issues = main()