"""FastAPI エンドポイントの単体テスト (scraper/generate_clip はモック)。"""

from __future__ import annotations

import time
from unittest.mock import patch

from fastapi.testclient import TestClient

from src import app as app_module
from src.models import SessionDetail, SpeakerInfo

client = TestClient(app_module.app)


def _detail(committee="内閣委員会"):
    return SessionDetail(
        chamber="shugiin", session_id="56328", date="2026-06-12",
        committee=committee, hls_url="https://x/y.m3u8", source_url="https://x",
        duration="1時間", speakers=[
            SpeakerInfo(name="山下貴司", affiliation="内閣委員長", role="委員長",
                        start_seconds=10.0, start_time="", duration_minutes=1),
            SpeakerInfo(name="高山聡史", affiliation="チームみらい", role="質疑者",
                        start_seconds=100.0, start_time="", duration_minutes=14),
        ],
    )


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_sessions_groups_by_committee():
    app_module._detail_cache.clear()
    with patch.object(app_module._scraper, "detect_new_sessions", return_value=["56328"]), \
         patch.object(app_module._scraper, "get_session_detail", return_value=_detail()):
        r = client.get("/api/sessions", params={"date": "2026-06-12"})
    assert r.status_code == 200
    data = r.json()
    assert len(data["sessions"]) == 1
    s = data["sessions"][0]
    assert s["committee"] == "内閣委員会"
    assert s["n_members"] == 1  # only 質疑者


def test_list_members_returns_only_questioners():
    app_module._detail_cache.clear()
    with patch.object(app_module._scraper, "get_session_detail", return_value=_detail()):
        r = client.get("/api/sessions/56328/members")
    assert r.status_code == 200
    members = r.json()["members"]
    assert len(members) == 1
    assert members[0]["name"] == "高山聡史"
    assert members[0]["affiliation"] == "チームみらい"


def test_create_clip_returns_job_and_completes():
    fake_result_path = app_module.MEDIA_DIR / "56328" / "高山聡史_clip.mp4"

    class FakeMember:
        name = "高山聡史"
        affiliation = "チームみらい"

    class FakeEDL:
        kept_duration = 120.0
        keep_ranges = [1, 2, 3]

    class FakeResult:
        clip_mp4 = fake_result_path
        member = FakeMember()
        edl = FakeEDL()

    def fake_generate(**kwargs):
        # call progress to exercise the callback
        if kwargs.get("progress"):
            kwargs["progress"]("scraping")
            kwargs["progress"]("rendering")
        fake_result_path.parent.mkdir(parents=True, exist_ok=True)
        return FakeResult()

    with patch.object(app_module, "generate_clip", side_effect=fake_generate):
        r = client.post("/api/clips", json={"session_id": "56328", "member": "高山聡史"})
        assert r.status_code == 200
        job_id = r.json()["job_id"]

        # poll until done (in-process executor; should be fast)
        for _ in range(50):
            jr = client.get(f"/api/jobs/{job_id}")
            assert jr.status_code == 200
            if jr.json()["state"] in ("done", "error"):
                break
            time.sleep(0.05)

    body = jr.json()
    assert body["state"] == "done", body
    assert body["result"]["member"] == "高山聡史"
    # 全体レンダ廃止: 生成結果に全体 clip は無い (プレビューはトピック単位オンデマンド)
    assert body["result"]["clip_path"] is None
    assert body["result"]["n_ranges"] == 3


def test_create_clip_reports_error_on_failure():
    def boom(**kwargs):
        raise RuntimeError("scrape failed")

    with patch.object(app_module, "generate_clip", side_effect=boom):
        r = client.post("/api/clips", json={"session_id": "999", "member": "誰か"})
        job_id = r.json()["job_id"]
        for _ in range(50):
            jr = client.get(f"/api/jobs/{job_id}")
            if jr.json()["state"] in ("done", "error"):
                break
            time.sleep(0.05)
    body = jr.json()
    assert body["state"] == "error"
    assert "scrape failed" in body["error"]


def test_job_not_found():
    assert client.get("/api/jobs/nonexistent").status_code == 404


def test_clip_file_not_found():
    assert client.get("/api/clips/file/56328/nope.mp4").status_code == 404


# ---------------------------------------------------------------------------
# /edit with qa_tree: ツリー選択 → ranges 再計算 → 再レンダ (rerender はモック)
# ---------------------------------------------------------------------------


def _project_with_tree(sent_enabled: list[bool]) -> dict:
    # 文 i は range i (i*10..i*10+10, member-WAV) に対応
    n = len(sent_enabled)
    ranges = [{"start": i * 10.0, "end": i * 10.0 + 10.0, "enabled": True}
              for i in range(n)]
    sentences = [{"text": f"文{i}", "start": i * 10.0, "end": i * 10.0 + 10.0,
                  "summary": "", "importance": "mid", "enabled": en}
                 for i, en in enumerate(sent_enabled)]
    return {
        "session_id": "56328", "member": "高山聡史",
        "source_video": "高山聡史_src.mp4", "member_start": 0.0,
        "aspect": "9:16", "subtitle_style": "plain", "title": "t",
        "ranges": ranges, "captions": [],
        "qa_tree": {"topics": [{"index": 0, "label": "", "question_speaker": "A",
                                "answer_speakers": [],
                                "turns": [{"speaker": "A", "role": "質疑者",
                                           "sentences": sentences}]}]},
    }


def _poll(job_id: str) -> dict:
    for _ in range(50):
        jr = client.get(f"/api/jobs/{job_id}")
        if jr.json()["state"] in ("done", "error"):
            return jr.json()
        time.sleep(0.05)
    return jr.json()


def test_edit_with_tree_recomputes_ranges():
    captured = {}

    def fake_rerender(project, out_dir, **kw):
        captured["enabled"] = [r.enabled for r in project.ranges]
        return out_dir / "高山聡史_clip.mp4"

    with patch.object(app_module, "rerender_project", side_effect=fake_rerender):
        body = _project_with_tree([True, False, True])  # 中央の文を外す
        r = client.post("/api/clips/56328/高山聡史/edit", json=body)
        assert r.status_code == 200
        out = _poll(r.json()["job_id"])
    assert out["state"] == "done", out
    # ツリーで中央 off → ranges も中央だけ無効に再計算
    assert captured["enabled"] == [True, False, True]
    assert out["result"]["n_ranges"] == 2


def test_edit_with_all_sentences_off_errors():
    with patch.object(app_module, "rerender_project",
                      side_effect=AssertionError("should not render")):
        body = _project_with_tree([False, False])  # 全 off
        r = client.post("/api/clips/56328/高山聡史/edit", json=body)
        assert r.status_code == 200
        out = _poll(r.json()["job_id"])
    assert out["state"] == "error"
    assert "選択が空" in out["error"]


def test_export_topics_returns_clip_list():
    def fake_topics(project, out_dir, **kw):
        return [
            {"clip_path": "56328/高山聡史_topic0_clip.mp4", "topic_index": 0,
             "topic_label": "トピックA", "duration": 12.3},
            {"clip_path": "56328/高山聡史_topic1_clip.mp4", "topic_index": 1,
             "topic_label": "トピックB", "duration": 8.0},
        ]

    with patch.object(app_module, "render_topic_clips", side_effect=fake_topics):
        body = _project_with_tree([True, True])
        r = client.post("/api/clips/56328/高山聡史/export?mode=topics", json=body)
        assert r.status_code == 200
        out = _poll(r.json()["job_id"])
    assert out["state"] == "done", out
    clips = out["result"]["clips"]
    assert len(clips) == 2
    assert clips[0]["topic_label"] == "トピックA"


def test_export_full_returns_single_clip():
    def fake_full(project, out_dir, **kw):
        return app_module.MEDIA_DIR / "56328" / "高山聡史_clip.mp4"

    with patch.object(app_module, "rerender_project", side_effect=fake_full), \
         patch.object(app_module, "render_topic_clips",
                      side_effect=AssertionError("topics should not run")):
        body = _project_with_tree([True, True])
        r = client.post("/api/clips/56328/高山聡史/export?mode=full", json=body)
        assert r.status_code == 200
        out = _poll(r.json()["job_id"])
    assert out["state"] == "done", out
    clips = out["result"]["clips"]
    assert len(clips) == 1
    assert clips[0]["topic_label"] == "全体"
    assert clips[0]["clip_path"].endswith("高山聡史_clip.mp4")
