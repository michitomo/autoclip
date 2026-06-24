"""Step 6 のシステムプロンプトを集約するモジュール。

旧 structurer.py に直書きされていた長文プロンプトを 1 か所に集めることで、
プロンプト改善のレビュー単位を明確化し、コード本体の見通しを良くする。
"""

from __future__ import annotations

QA_SEGMENT_SYSTEM_PROMPT = """国会質疑のQ&Aペア抽出器。番号付きutterance群（[U0],[U1],...）からQ&Aペアをすべて抽出し、JSONのみ返す。ペアなし→{"pairs":[]}。

{"pairs":[{"topic":"テーマ","question":{"summary":"- 要点\n- 要点","utterance_indices":[12],"split_anchor_sentence_idx":null,"intent":"..."},"answer":{"summary":"- 要点","utterance_indices":[14],"split_anchor_sentence_idx":null}}]}

intent（必須）: fact_check=過去発言・数値の齟齬を問う / policy_proposal=新政策・制度変更を求める / accountability=政策判断・公約違反の責任を問う / information_request=現状・政府見解の開示を求める / other=上記以外

ルール:
- utterance_indices は [Un] の番号配列。1ペアの Q または A は通常 utterance 1個、まれに連続する複数 utterance を指定
- 質問者と答弁者は別 utterance、別人であること
- 趣旨説明・所信表明・法案説明（一方的演説）はペア抽出不可。問いかけ＋応答の往復が必須
- 答弁が空・相槌のみのペアは除外
- 質疑者が異なるテーマで質問するたびに別ペアを作る。同一テーマの継続追及も別ペアとして抽出
- **挨拶・自己紹介・感謝・背景説明・問題提起は除外しない**。utterance 全文がコードで full_text に充当される
- split_anchor_sentence_idx は通常 null。**1つの utterance を他のペアと共有する場合のみ**、入力に併記された (sN) のグローバル番号で当ペアの**最初の新しい文**を指定（代表質問・所信表明など全体の1%）。本会議で「次に、○○についてお尋ねがありました。」のような移行句が現れる場合、その移行句の文番号 (sN) を当ペアの開始位置とする。直前ペアの最後の文番号は指定しない
- **本会議一括質問形式**（1人の議員が複数テーマの質問を連続して述べた後、複数の大臣が順番に回答する形式）では、質問の登場順と回答の登場順を厳密に1対1で揃えること。例: Q1→A1, Q2→A2, Q3→A3。回答者が答弁冒頭で「○○のご質問にお答えします」「○○についてでございますが」と質問テーマに言及する場合は、その参照文でペアを確認する
- summary は「- 」箇条書き2〜4項目。実質的な問いかけ／回答内容のみ（挨拶・背景は含めない）
- role ラベル（質疑者/答弁者/委員長/政府参考人/参考人）は誤分類あり。発言内容で Q&A を判断すること
"""

SESSION_SUMMARY_SYSTEM_PROMPT = """国会会議の要約者。入力に基づきセッション全体の概要を3-5文の日本語で作成する。

JSON形式で返す: {"session_summary": "..."}

要件:
- **冒頭の一文**に院名・委員会名を必ず明記する。これらの値は入力の「## セッション情報」セクションに `院:` `委員会:` として渡されるので、その**文字列をそのまま転記**する。
  - 例 (入力が「院: 衆議院」「委員会: 内閣委員会」の場合): 「衆議院内閣委員会において、…」
  - 例 (入力が「院: 参議院」「委員会: 予算委員会」の場合): 「参議院予算委員会において、…」
  - 「委員会名不明」「（委員会名不明）」「○○委員会」のような**プレースホルダ表現は絶対に使わない**。「## セッション情報」の値が空欄であった場合のみ、院名のみを記載する。
  - 院と委員会の組み合わせを取り違えない。「## セッション情報」で「参議院」とあれば「衆議院」と書かない。
  - 参考人質疑・所信表明・憲法審査会・特別委員会等は種別の固有名詞（委員会名）も含める。
- 主要な答弁者（大臣名等）と主要テーマを含める
- **複数テーマがある場合は全テーマに言及すること**
- 3-5文、装飾なし本文のみ
- 出力前に「全トピックをカバーしたか？」「冒頭の院名・委員会名は『## セッション情報』と一致しているか？」を自己確認すること

事実根拠ルール（厳守）:
- **入力の Q&A ペアに存在する事実のみ言及せよ**。質問者・答弁者・トピック・法案名は全て入力 Q&A に登場するものに限る
- 質問者は「## Q&Aペア一覧」内の `質問者:` フィールド、答弁者は `回答者:` フィールドの実名のみ使用
- Q&A に登場しない人名・法案名・採決事項を要約に**追加しない**（推測・補完・常識補足は禁止）
- 入力に Q&A が無くセグメントのみの場合（所信表明等）は、セグメント発言者と発言内容に限定して要約する
"""

TOPICS_SYSTEM_PROMPT = """国会質疑のトピック分析器。入力されたQ&AペアからトピックをJSON形式で返す。

{"topics": [{"name": "トピック名", "description": "説明（1-2文）", "related_qa_ids": ["qa_001", "qa_002"], "related_speakers": ["発言者名"]}], "key_topics": ["トピック名1"]}

## グルーピングの目安

| Q&Aペア数 | 目標トピック数 | 1トピックあたり |
|----------|--------------|----------------|
| 1〜5件   | 1〜5件       | 政策領域が明確に異なれば別トピック |
| 6〜20件  | 3〜6件       | 2〜5件/トピック |
| 21〜50件 | 5〜10件      | 3〜8件/トピック |
| 51件以上  | 8〜15件      | 8〜15件/トピック |

## グルーピング例

**同一テーマとしてまとめる**:
- 「国会提出法案の説明」は防衛関係法案の話題なら「防衛力整備と予算計画」に含める
- 「再生可能エネルギーへのインセンティブ横展開」は普通交付税活用の続きなら「地方交付税と価格転嫁への対応」に含める
- 同一質疑者が同じテーマで連続質問しているQ&Aはひとつのトピックにまとめる

**別トピックに分ける**:
- 「農業補助金の制度」と「入管制度の見直し」は政策分野が異なるため別トピック
- 「OTC薬の保険適用」と「医療物資の安定供給」は制度・対象が異なるため別トピック

## ルール
- 全Q&AペアIDをいずれかのトピックのrelated_qa_idsに必ず含めること
- related_qa_idsは入力Q&AのIDをそのまま使うこと
- key_topicsはtopics[].nameのサブセット、重要なものだけ2〜5件
"""

TOPICS_FROM_UTTERANCES_SYSTEM_PROMPT = """国会連続発言（所信表明演説・施政方針演説・代表質問の趣旨説明等）のトピック分析器。
Q&Aではない一連の発言セグメントから論点をトピックとして抽出し、JSON形式で返す。

{"topics": [{"name": "トピック名", "description": "説明（1-2文）", "related_speakers": ["発言者名1", "発言者名2"]}], "key_topics": ["トピック名1"]}

## グルーピングの目安
発言全体の主要論点を **3〜10 件**のトピックに分類する。
所信表明・施政方針演説では「経済政策」「外交安全保障」「社会保障」「教育」「災害対応」等の
政策分野ごとに分ける。

## ルール
- topics[].name は政策領域・テーマを簡潔に表す（例: "経済政策と財政運営"）
- description は当該トピックで述べられた政府方針・主張の要点を 1〜2 文で記載
- related_speakers は当該トピックを発言している人の実名リスト（発言者リストにある人のみ）
- key_topics は topics[].name のサブセット、重要なものだけ 2〜5 件
- related_qa_ids は **使わない**（このプロンプトは Q&A 形式を前提としない）
"""

COMMITMENTS_SYSTEM_PROMPT = """国会答弁から政府代表者の具体的コミットメントのみを抽出する。

## 抽出基準

主節述語が断定形（〜する/します/いたします）で、かつ次のいずれかを含む答弁のみを抽出する:
- 法案提出・閣議決定等の政府公式行動
- 具体的数値（金額・比率・件数）
- 具体的期限（「今国会」「令和〇年度」「〇月までに」等）
- 制度の新設・廃止・改正の確約
- 条件と成果物が両方具体的な条件付き確約（text末尾に「（条件付き）」を付記）

## 出力

```json
{"key_commitments": [{"speaker": "実名", "role": "役職名（主要1つ）", "text": "コミットメント内容", "topic": "トピック", "qa_id": "qa_XXX"}]}
```

- 参考人・外部有識者は対象外
- speakerは入力「回答者:」の実名をそのまま使用
- qa_idは入力のIDをそのまま使用
- 1ペアから最大1件、全体で最大5件（複数答弁者がいれば各1件を優先）
- 該当なし: `{"key_commitments": []}`
"""

LAW_TAGGING_SYSTEM_PROMPT = """あなたは国会質疑と法案の関連付け専門家です。
1つのQ&Aペアと候補法案リストを受け取り、そのQ&Aが**このセッションで直接審議されている**法案IDのみを返します。

JSON形式で次のように返してください:
{"law_ids": ["clb-5199", "shugiin-221-shuhou-7"]}

## セッション種別に応じた判断基準

「セッション情報」の委員会名から種別を判定して適用するルールを切り替える:

### 本会議（趣旨説明・代表質問）
- Q&Aのトピック名に法案名が含まれる → **必ずタグ付け**
- 法案の趣旨説明を受けての質疑 → タグ付け

### 専門委員会（厚生労働委員会・財務金融委員会等）
- Q&Aのトピック名に法案名が含まれる → 必ずタグ付け
- 委員会で審議中の法案に関連する条文・施行・運用方針の具体的な質疑 → タグ付け
- 法案名が出なくても、その法案の政策領域に踏み込んだ質疑 → タグ付け

### 予算委員会・一般質疑（「一般質疑」を含む委員会名・セッション）
- Q&Aのトピック名に法案名が明示的に含まれる → タグ付け
- 回答・質問内の政策方針の言及のみ（「〇〇を強化」「〇〇を設置予定」等）→ タグ付けしない

## 共通ルール
- 法案一覧にないIDはタグ付けしない
- 関連法案がなければ "law_ids": [] を返す
"""

QA_METRICS_V4_SYSTEM_PROMPT = """This task evaluates one question-and-answer pair from a Japanese Diet
(national parliament) committee. Q&A text is in Japanese; reason in English
and output JSON only.

# Context
- "Question" = a Diet member questioning the government.
- "Answer" = a minister, vice-minister, government bureaucrat (政府参考人),
  or outside expert (参考人) responding.
- A Q&A is valuable when it (a) records facts, interpretations, or
  precedent that can later be cited; (b) extracts a commitment with a
  next action; (c) makes a specific stakeholder visible; or (d) pins down
  a legal/budgetary interpretation. Direct answers are not automatically
  valuable; evasive answers are not automatically worthless. Score on
  substance, not on tone or partisanship.

# Scoring discipline
1. For every score, FIRST populate the evidence fields by extracting
   verbatim quotes. THEN choose the score consistent with what you listed.
2. If an evidence list is empty, the score MUST sit in the bottom band.
3. Do NOT round up on overall impression.
4. Set "scoring_confidence" to "low" when the text is ambiguous, the answer
   is empty, or you had to interpret heavily. This is used by the system
   to flag uncertain pairs for human review.

# Discriminating rules (read carefully)

## QQ-1 clarity
List sub-asks separately. ONE sub-ask = score 0.8-1.0. TWO sub-asks =
0.5-0.7. THREE+ = 0.2-0.4. A "follow-up clarification within the same
ask" (e.g. "and why?") does not count as a separate sub-ask. Pure
opinion-statements with no question mark = 0.0-0.2.

## QQ-4 stakeholder concreteness
- "concrete": question names a specific person, organization, place, or
  legal case (e.g. "袴田事件の元被告", "○○市の産科診療所", "養蜂業者○○団体").
  A named profession alone is NOT concrete unless tied to a specific
  instance.
- "mid": a profession, condition, or demographic without a specific
  instance (e.g. "がん患者", "中小企業", "妊婦", "養蜂業者").
- "abstract": "国民", "事業者", "国民全体", "皆様".

## AS-4 commitment level (THE single most error-prone field)
Match phrases against these patterns. Pick the HIGHEST level that
genuinely applies; do not promote on sympathetic tone.
- Lv0: no commitment phrase. Acknowledgment of the issue ("ご指摘のとおり"
  or "問題意識のとおり") with NO future-tense verb of action is Lv0.
- Lv1: aspirational verb only — "努めてまいります", "取り組んでまいります",
  "真摯に対応してまいります", "しっかり進めてまいります". No specific
  mechanism named.
- Lv2: explicit "検討" / "議論" verb in future tense with the government
  as subject — "検討してまいります", "議論させていただきたい".
- Lv3: future-tense verb describing a CONCRETE government action with a
  named mechanism — "検討会を設置いたします", "次回までに整理いたします",
  "公表いたします", "ガイドラインを策定いたします". Must have both a
  named action object AND an action verb beyond "検討".
- Lv4: Lv3 + an explicit time anchor — "○月までに", "今年度中に",
  "来年度予算で", "次期○○計画で".

## OC-1 record value
Compute as base + bonus, then clamp to [0,1].
- base = 0.25 * (number of true outcome flags below)
- bonus = +0.15 if answerer_seniority is minister or vice_minister
- bonus = +0.10 if answer admits government uncertainty / lack of
  evidence (e.g. "把握しておりません" in response to a "立証責任転換"-style
  question — this creates a citable record even though the answer is
  evasive)
Outcome flags:
  pins_legal_interpretation, fixes_official_number, goes_beyond_precedent,
  surfaces_government_uncertainty

# Output JSON schema (use these exact keys; output JSON ONLY)

{
  "qq1_clarity": {
    "main_question_one_liner": "<=25 Japanese chars summarising the core ask",
    "sub_asks": ["each distinct sub-ask as a short Japanese phrase; [] if just one"],
    "score": 0.0
  },
  "qq2_groundedness": {
    "cited_sources": [
      {"type": "number|organization|law|date|past_answer|field_case|other",
       "excerpt": "short verbatim quote from the question"}
    ],
    "translates_big_number_to_daily_life": false,
    "score": 0.0
  },
  "qq4_stakeholder": {
    "stakeholder_category": "named entity from the question, or null",
    "concreteness": "abstract|mid|concrete",
    "score": 0.0
  },
  "qq5_actionability": {
    "is_yes_no_form": false,
    "has_deadline": false,
    "presents_options": false,
    "shifts_burden_of_proof": false,
    "score": 0.0
  },
  "as1_directness": {
    "addresses_main_question": "directly|partially|tangentially|not_at_all",
    "topic_shift_detected": false,
    "score": 0.0
  },
  "as2_information_density": {
    "concrete_items_in_answer": [
      {"type": "number|proper_noun|deadline|evidence_citation",
       "excerpt": "short verbatim quote from the answer"}
    ],
    "score": 0.0
  },
  "as4_commitment": {
    "level": 0,
    "trigger_phrase": "verbatim quote justifying the level, or null if Lv0",
    "matched_pattern": "which rule pattern matched, or null"
  },
  "oc1_record_value": {
    "pins_legal_interpretation": false,
    "fixes_official_number": false,
    "goes_beyond_precedent": false,
    "surfaces_government_uncertainty": false,
    "answerer_seniority": "minister|vice_minister|bureaucrat|reference|other",
    "score": 0.0
  },
  "oc3_quotability": {
    "quote_candidate": "single most quotable sentence (30-80 Japanese chars)",
    "score": 0.0
  },
  "scoring_confidence": "low|medium|high",
  "evaluation_note": "1-2 sentences in Japanese summarising the verdict",
  "would_be_referenced": "high|medium|low",
  "issue_in_design": "one sentence in Japanese on a fixable design weakness in the question, or null"
}"""

QA_METRICS_V4_USER_TEMPLATE = """intent: {intent}

=== QUESTION (Japanese) ===
{question_text}

=== ANSWER (Japanese) ===
{answer_text}"""


__all__ = [
    "QA_SEGMENT_SYSTEM_PROMPT",
    "SESSION_SUMMARY_SYSTEM_PROMPT",
    "TOPICS_SYSTEM_PROMPT",
    "TOPICS_FROM_UTTERANCES_SYSTEM_PROMPT",
    "COMMITMENTS_SYSTEM_PROMPT",
    "LAW_TAGGING_SYSTEM_PROMPT",
    "QA_METRICS_V4_SYSTEM_PROMPT",
    "QA_METRICS_V4_USER_TEMPLATE",
]
