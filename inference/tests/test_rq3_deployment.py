"""Small synthetic unit fixtures only; never used as experimental predictions."""
import copy
import csv
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]
import rq3_deployment as rq3


class RQ3Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.row={"gitreq_id":"test-1","requirement_text":"Synthetic fixture", "binary_label":"Security",
                  "original_label":"Security","repository":"test/repo","url":"https://example.invalid/1"}
        self.route={"route_id":"model__hash", "model":"m","prompt_sha256":"p","route_fingerprint_sha256":"r"}
        self.ctx={"gitreq_id":"test-1","rendered_request_sha256":"req","input_tokens":10}
        self.vote={"raw":"Security","label":"Security","finish_reason":"stop","input_tokens":10,
                   "rendered_request_sha256":"req"}
        self.rec={"gitreq_id":"test-1","model":"m","prompt_sha256":"p","route_id":"model__hash",
                  "route_fingerprint_sha256":"r","requirement_text_sha256":rq3.sha("Synthetic fixture"),
                  "gold_binary":"Security","original_label":"Security","repository":"test/repo",
                  "url":"https://example.invalid/1","votes":[self.vote],"prediction":"Security"}

    def test_projection_uses_first_not_majority(self):
        rec=copy.deepcopy(self.rec)
        rec["votes"]=[dict(self.vote,raw="Non-security",label="Non-security"),self.vote,self.vote]
        out=rq3.project_first_vote(rec,self.route,{"test":True})
        self.assertEqual(out["prediction"],"Non-security")
        self.assertEqual(rec["prediction"],"Security")
        self.assertEqual(len(out["votes"]),1)

    def test_invalid_and_truncated_rejected(self):
        for changes in [{"label":"INVALID"},{"raw":"Label: Security"},{"finish_reason":"length"},
                        {"rendered_request_sha256":"wrong"},{"input_tokens":12}]:
            with self.subTest(changes=changes), self.assertRaises(AssertionError):
                rq3.check_vote(dict(self.vote,**changes),self.ctx)

    def test_cross_model_record_rejected(self):
        with self.assertRaises(AssertionError):
            rq3.check_row(dict(self.rec,model="other"),self.row,self.route,self.ctx)

    def test_resume_repairs_only_torn_json(self):
        p=self.root/"partial.jsonl"
        p.write_text(json.dumps(self.rec)+"\n"+'{"gitreq_id":')
        self.assertEqual(rq3.resume_prefix(p,[self.row]*2,self.route,[self.ctx]*2),1)
        self.assertEqual(len(list(self.root.glob("*.torn-*"))),1)
        self.assertEqual(len(rq3.jl(p)),1)

    def test_resume_does_not_drop_semantic_corruption(self):
        p=self.root/"partial.jsonl"
        p.write_text(json.dumps(dict(self.rec,model="wrong")))
        with self.assertRaises(AssertionError):
            rq3.resume_prefix(p,[self.row],self.route,[self.ctx])
        self.assertEqual(len(list(self.root.glob("*.torn-*"))),0)

    def test_resume_complete_unterminated_line(self):
        p=self.root/"partial.jsonl"; p.write_text(json.dumps(self.rec))
        self.assertEqual(rq3.resume_prefix(p,[self.row],self.route,[self.ctx]),1)
        self.assertTrue(p.read_bytes().endswith(b"\n"))

    def test_completion_checks_hash(self):
        spec={"output_root":str(self.root)}
        final,partial,marker=rq3.paths(spec,self.route)
        partial.parent.mkdir(parents=True)
        partial.write_text(json.dumps(self.rec)+"\n")
        rq3.finish(spec,self.route,[self.row],[self.ctx],{"type":"test"})
        self.assertTrue(rq3.completed(spec,self.route,[self.row],[self.ctx]))
        final.write_text(final.read_text()+" ")
        with self.assertRaises(AssertionError):
            rq3.completed(spec,self.route,[self.row],[self.ctx])

    def test_fingerprint_keeps_template_and_hardware_not_index(self):
        a={"local_artifact":{"revision_metadata_files":4,"weight_index_sha256":"weights"},
           "cuda_visible_devices":"0","logical_gpu_name":"V100"}
        self.assertEqual(rq3.fingerprint(a),rq3.fingerprint(dict(a,cuda_visible_devices="1",chat_template_kwargs={})))
        self.assertNotEqual(rq3.fingerprint(a),rq3.fingerprint(dict(a,chat_template_kwargs={"enable_thinking":False})))
        self.assertNotEqual(rq3.fingerprint(a),rq3.fingerprint(dict(a,logical_gpu_name="other")))

    def test_unique_routes_preserve_all_logical_cells(self):
        models=["m1","m2"]
        spec={"model_order":models,"models":{m:{"interface":"appendix_f","stop_strings":[]} for m in models},
              "runtime":{},"gitreq_enriched_sha256":"dataset"}
        runtime={m:{"local_artifact":{},"logical_gpu_name":"V100"} for m in models}
        def p(name,text):
            return {"prompt_id":name,"semantic_text":text,"semantic_sha256":rq3.sha(text)}
        routes,logical=rq3.build_manifests(spec,{"p0":p("p0","a"),
                            "prompts":[p("cell1","a"),p("cell2","b"),p("cell3","b")]},runtime)
        self.assertEqual(len(routes),4)
        self.assertEqual(len(logical),8)
        self.assertEqual(len({r["route_fingerprint_sha256"] for r in routes}),4)
        self.assertEqual(sum(not l["is_baseline"] for l in logical),6)

    def test_csv_union_columns(self):
        path=self.root/"test.csv"
        rq3.csv_write(path,[{"name":"p0"},{"name":"pstar","iteration":20}])
        with path.open() as f:
            self.assertEqual(list(csv.DictReader(f))[1]["iteration"],"20")

    def test_known_metrics(self):
        m=rq3.compute_metrics(["Security","Non-security","Non-security","Non-security"],
                             ["Security","Security","Non-security","Non-security"],rq3.LABELS).as_dict()
        self.assertAlmostEqual(m["macro_f1"],(2/3+.8)/2)
        self.assertAlmostEqual(m["per_class"]["Security"]["recall"],.5)

    def test_legacy_rq1_schema_restores_runtime_with_source_hash(self):
        runtime={"local_artifact":{},"model_name":"m"}
        route={**self.route,"contract":{"target_sha256":"data","model_fingerprint":rq3.fingerprint(runtime)}}
        manifest={"gitreq_enriched_sha256":"data","source_archive_sha256":"archive",
                  "votes_per_item":3,"target_rows":6302,"model_fingerprints":{"m":runtime},
                  "routes":[{"deployer":"m","prompt_id":"p0","semantic_sha256":"p"}]}
        completion={"prediction_sha256":"filehash","model":"m","prompt_sha256":"p","rows":6302,"votes_per_item":3}
        marker={"prediction_source":"rq1_reused","source_route_id":"p0","source_prediction_sha256":"filehash","source_run_id":"old-run"}
        original={**self.rec,"semantic_sha256":"p"}
        spec={"rq1_root":str(self.root),"source_archive_sha256":"archive"}
        with patch.object(rq3,"read_json",side_effect=[manifest,completion]), patch.object(rq3,"sha_file",return_value="filehash"), patch.object(rq3,"jl",return_value=[original]*6302):
            evidence=rq3.legacy_rq1_evidence(spec,marker,route,[self.rec]*6302)
            self.assertEqual(evidence["source_run_id"],"old-run")
        with patch.object(rq3,"read_json",return_value=manifest), patch.object(rq3,"sha_file",return_value="wrong"), self.assertRaises(AssertionError):
            rq3.legacy_rq1_evidence(spec,marker,route,[self.rec]*6302)


if __name__=="__main__":
    unittest.main()
