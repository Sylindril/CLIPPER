#!/usr/bin/env python3
"""download.py - fetch the raw dataset sources for test_suite/, tuning_suite/ and text_suite/,
verify them byte-for-byte against this repo's original experiments, and run the
repo's existing converters to regenerate the *.json files.

We don't redistribute the datasets themselves (license terms vary), so this
script downloads each raw file from its public source, checks its sha256
against the exact bytes our experiments used, and then hands off to
``src/tools/build_{test,tuning,text}_suite.py`` -- the
same converters that produced the committed JSON. After conversion it checks
the resulting JSON against a second sha256 table, so a clean run either
reproduces the committed data exactly or tells you precisely where it diverged.

One development dataset -- ``move1`` -- has no public URL, so its three raw
files ship in tuning_suite/; they are verified and converted like the rest.

Usage::

    python datasets/download.py                 # all suites
    python datasets/download.py --suite test     # test_suite/ only
    python datasets/download.py --suite tuning    # tuning_suite/ only
    python datasets/download.py --suite text      # text_suite/ only
    python datasets/download.py --suite recent    # recent_suite/ only
    python datasets/download.py --no-verify       # skip sha256 checks (download anyway)

Only the standard library is used (urllib, hashlib, zipfile). Converters are
run with the same Python interpreter running this script.
"""

import argparse
import hashlib
import io
import os
import subprocess
import sys
import urllib.request
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER_AGENT = "Mozilla/5.0 (dataset-download-script; +https://archive.ics.uci.edu)"
TIMEOUT = 60

# ---------------------------------------------------------------------------
# Raw source files: where each one comes from, and the sha256 of the exact
# bytes committed in this repo (i.e. the bytes our experiments used).
#
# fetch types:
#   ("http", url)                          -- plain file download
#   ("zip", url, member)                   -- download a zip, extract one member
#   ("zipzip", url, outer_member, member)  -- a zip containing another zip
#   ("manual",)                            -- shipped with this repo; verify only
# ---------------------------------------------------------------------------

RAW_FILES = [
    # --- test_suite/ --------------------------------------------------
    dict(suite="test", dest="test_suite/adult.csv",
         sha256="5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
         fetch=("http", "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")),
    dict(suite="test", dest="test_suite/adult.names",
         sha256="c248284c0b5de30c9e1958d6cdd168a34a654758b620e68f46aefa83fc0a576a",
         fetch=("http", "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names")),
    dict(suite="test", dest="test_suite/bank_marketing.csv",
         sha256="d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d",
         fetch=("zip", "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip",
                "bank-full.csv")),
    dict(suite="test", dest="test_suite/bank_marketing.names",
         sha256="3899a77082573ed676a7dfce3765a5307915c7f11b1fe2f6f55fd3c5467c12f7",
         fetch=("zip", "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip",
                "bank-names.txt")),
    dict(suite="test", dest="test_suite/college.csv",
         sha256="2e6fd75a63416328cd5d76cf829673315bec9c83b0a71141175892cdfac57031",
         fetch=("http", "https://raw.githubusercontent.com/cdiegor/MineracaoDeDados/main/gotocollege.csv")),
    dict(suite="test", dest="test_suite/water_potability.csv",
         sha256="904004bde729bfe3d2e195f46343bceead09e32a0eb95bb8184e7e20e029b2bf",
         fetch=("http", "https://raw.githubusercontent.com/MainakRepositor/Datasets/master/water_potability.csv")),
    dict(suite="test", dest="test_suite/stroke.csv",
         sha256="644d473b05d2797006bd94865e4f8bb057f0c721617911613c82c8fcfc707420",
         fetch=("http", "https://raw.githubusercontent.com/JulioArapa/mypackage/main/healthcare-dataset-stroke-data.csv")),
    dict(suite="test", dest="test_suite/telco_churn.csv",
         sha256="16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91",
         fetch=("http", "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv")),
    dict(suite="test", dest="test_suite/compas.csv",
         sha256="c451db85908b2f7fef1d83203bedf6b71ecda0d5af468d82ae62178f91d0cc7d",
         fetch=("http", "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv")),
    dict(suite="test", dest="test_suite/heloc.csv",
         sha256="e5a914f742ad7f8472c5417902ec06b321ebc5de1d68ec13f136d62c87a7fbf3",
         fetch=("http", "https://raw.githubusercontent.com/benoitparis/explainable-challenge/master/heloc_dataset_v1.csv")),

    # --- tuning_suite/ --------------------------------------------------
    dict(suite="tuning", dest="tuning_suite/hypothyroid.data",
         sha256="ffba466e9844cf7d6d8fde1f89209c6ab1d583c865abb449bb6ab8c70875f4c4",
         fetch=("http", "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hypothyroid.data")),
    dict(suite="tuning", dest="tuning_suite/hypothyroid.names",
         sha256="686ba47a937f812e70240eab449cbe571bbd060ce49424dc7bd7d325ed8cd690",
         fetch=("http", "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hypothyroid.names")),
    dict(suite="tuning", dest="tuning_suite/breast_cancer_wisconsin.data",
         sha256="402c585309c399237740f635ef9919dc512cca12cbeb20de5e563a4593f22b64",
         fetch=("http", "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data")),
    dict(suite="tuning", dest="tuning_suite/cylinder_bands.data",
         sha256="31b4bd7f6cbca567e06199ae8bbe085c00d64657fcb2daad0c214cb4d08076f7",
         fetch=("http", "https://archive.ics.uci.edu/ml/machine-learning-databases/cylinder-bands/bands.data")),
    dict(suite="tuning", dest="tuning_suite/abalone.data",
         sha256="de37cdcdcaaa50c309d514f248f7c2302a5f1f88c168905eba23fe2fbc78449f",
         fetch=("http", "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data")),
    dict(suite="tuning", dest="tuning_suite/german.data",
         sha256="b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871",
         fetch=("http", "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")),
    dict(suite="tuning", dest="tuning_suite/pima_diabetes.data",
         sha256="6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af",
         fetch=("http", "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv")),
    dict(suite="tuning", dest="tuning_suite/titanic.csv",
         sha256="4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7",
         fetch=("http", "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")),
    dict(suite="tuning", dest="tuning_suite/obesity.csv",
         sha256="28b6991062855d3d15642e38b1aa9d1b32fb7abbb5abf5b096f1868cd706154b",
         fetch=("zip",
                "https://archive.ics.uci.edu/static/public/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition.zip",
                "ObesityDataSet_raw_and_data_sinthetic.csv")),
    dict(suite="tuning", dest="tuning_suite/student-por.csv",
         sha256="a7594a11d7771c0efe1a740824e0e833da9c4cad07c39a9766a874575563fb3f",
         fetch=("zipzip", "https://archive.ics.uci.edu/static/public/320/student+performance.zip",
                "student.zip", "student-por.csv")),
    # move1: from the original RIPPER/SLIPPER distribution; no public URL, so
    # the raw files ship in tuning_suite/.
    dict(suite="tuning", dest="tuning_suite/move1.data",
         sha256="cd066b46b41a65cfbca73114bed9e06be9fb2b1bd624dc654567b770735dbc0c",
         fetch=("manual",)),
    dict(suite="tuning", dest="tuning_suite/move1.names",
         sha256="af44ad8e6c018114ae21703ea25bf774abff64836a79cb29470058d0f2bef570",
         fetch=("manual",)),
    dict(suite="tuning", dest="tuning_suite/move1.test",
         sha256="3a7f7505148bb59b37a595cf7451fd0ce348591fc2490c5e4ee0f95ac09e48dd",
         fetch=("manual",)),

    # --- text_suite/ (UCI; raw files under text_suite/raw/) ---------------
    dict(suite="text", dest="text_suite/raw/sms/SMSSpamCollection",
         sha256="7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d",
         fetch=("zip", "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
                "SMSSpamCollection")),
    dict(suite="text", dest="text_suite/raw/sentiment/amazon_cells_labelled.txt",
         sha256="47003fc0a0d4840b00e96e715b6189bad09e7443a3da41c4cbe12ffc79f86ae3",
         fetch=("zip", "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip",
                "sentiment labelled sentences/amazon_cells_labelled.txt")),
    dict(suite="text", dest="text_suite/raw/sentiment/imdb_labelled.txt",
         sha256="aef2e49e3da25714d61175e3a6e68eeef74a20a2f914318dc3be9947ea86512d",
         fetch=("zip", "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip",
                "sentiment labelled sentences/imdb_labelled.txt")),
    dict(suite="text", dest="text_suite/raw/sentiment/yelp_labelled.txt",
         sha256="c76468b7b5c6e56a0804d728345c5f84aa2142ddb214420f61cc9cfd4c00d2ea",
         fetch=("zip", "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip",
                "sentiment labelled sentences/yelp_labelled.txt")),
    dict(suite="text", dest="text_suite/raw/youtube/Youtube01-Psy.csv",
         sha256="19797e6c77690e3c8809cfd2853ae7341390636367ba66cf5d4f4083f0b88535",
         fetch=("zip", "https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip",
                "Youtube01-Psy.csv")),
    dict(suite="text", dest="text_suite/raw/youtube/Youtube02-KatyPerry.csv",
         sha256="902c614f8ef24f987d6f614d7e6111aa5160b89a0646b68e007bd6044a3d123b",
         fetch=("zip", "https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip",
                "Youtube02-KatyPerry.csv")),
    dict(suite="text", dest="text_suite/raw/youtube/Youtube03-LMFAO.csv",
         sha256="702ef589860a1831956f527760a3d9737ef8a07ab36c7de35b92b8898b8c3928",
         fetch=("zip", "https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip",
                "Youtube03-LMFAO.csv")),
    dict(suite="text", dest="text_suite/raw/youtube/Youtube04-Eminem.csv",
         sha256="92f54eb6b22fdf3b7ae85e1f500e5aa7442edd025e504b988a97078756187e76",
         fetch=("zip", "https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip",
                "Youtube04-Eminem.csv")),
    dict(suite="text", dest="text_suite/raw/youtube/Youtube05-Shakira.csv",
         sha256="1d8ab47b71e8037c51183b2fc62f0591a48a4b54f3a4f5d9d3043113b274e98e",
         fetch=("zip", "https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip",
                "Youtube05-Shakira.csv")),
    # --- recent_suite/ (UCI 1125, published after the model's training cutoff) ---
    dict(suite="recent", dest="recent_suite/raw/inflation/classified_abstracts.json",
         sha256="be99f22556d1b38115dd1e1d96c902c465133f044d9c9bb0a41ae53f9742b8d3",
         fetch=("zip", "https://archive.ics.uci.edu/static/public/1125/inflation+research+abstracts+classification.zip",
                "classified_abstracts.json")),
]

MOVE1_NOTE = """\
tuning_suite/move1.{data,names,test} ship with this repository (move1 has no
public download).  Restore them from version control if they are missing.
"""

# ---------------------------------------------------------------------------
# Expected sha256 of the converted JSON files (what build_test_suite.py /
# build_tuning_suite.py should produce from the raw files above).
# ---------------------------------------------------------------------------

EXPECTED_JSON = {
    "recent": {
        "recent_suite/inflation_abstracts.json": "ffee1926b4edfabc77b2f2bbda9fe9d72f971f5f7a04cd89dea711e3215dea03",
    },
    "text": {
        "text_suite/sentiment_amazon.json": "cf57d29597f90b094fdec2a998355813ddd30a7be4ac6d30cb81a53089ebd67f",
        "text_suite/sentiment_imdb.json": "079488461f1912a438d0787b7112d8727bcf0eef36dcd2e897338740404daa22",
        "text_suite/sentiment_yelp.json": "3222fc7d94d20150dcfa3f660b4665f63dbb6bda1c6e2118410ee22a8fcc59af",
        "text_suite/sms_spam.json": "6b8f92bed7ae003f776fd289416a136a9cfc7c21d3a2e2a52daf2f0b9008ba15",
        "text_suite/youtube_spam.json": "163a44e65c75c77ba9f9d2d3f4e1e26db7961c164d22d0382264252724c95d96",
    },
    "test": {
        "test_suite/adult.json": "38429dc63019492339f71363ebe4bd493cd9df44362cdfb0ed1c57b28e157500",
        "test_suite/bank_marketing.json": "428cb1e4d20f14ca49ef47c13dded9c4b0891087baa355e0d11b59ecda420b18",
        "test_suite/college.json": "634e99305338e1fdc4d031257a946cd93e00a8108bca607cbcbb46e5f4575a1d",
        "test_suite/compas.json": "15587ef6ee1bf7a9b0287038816a1b135ff2e9a389a2e9057cb8dfa0f8fd59c3",
        "test_suite/heloc.json": "b007b8efa1f101e98f54fd9aee7bc1f590da25c11a701b1285991437d6f228d4",
        "test_suite/stroke.json": "8c73c44b16639d545a95cac5e7f41282690ec889316a1cee95b135819979de37",
        "test_suite/telco_churn.json": "ef0ab5a8cbc15175e0a21f36bf9bedc89f7b35d9d2444a35cdaf0dae2321d830",
        "test_suite/water_potability.json": "a5fc17c6272de762c960682661a186a2e685787e38f64dd1e7acab11b17558d4",
    },
    "tuning": {
        "tuning_suite/abalone.json": "85200cab85a465d3804105f88a8febc7fce4bb5804ebc8ac29a215b3629358c8",
        "tuning_suite/breast_cancer_wisconsin.json": "ad71e91498abc79fa899991bf7c1755582d66f7ff73b4270cb140659d5ab3a28",
        "tuning_suite/cylinder_bands.json": "0a72c56c4a13a68d942aa6c23dcc4a490101cfaef4498fb05d08bac93c927d94",
        "tuning_suite/german_credit.json": "7ee547a7e69306ded99a072d524ef4cd59e51fd412e367a5158ddde19c4b06a6",
        "tuning_suite/hypothyroid.json": "75e8692794337dce9a118035f4acd9f0ecd7fcfd877a6d1d0b18c32feb0c68f5",
        "tuning_suite/move1.json": "48117a17e49ebc838940414dc65b47150a562a10eb2c01168c4549df5c895a61",
        "tuning_suite/obesity.json": "112772a91ec3128399acbd74785c326597b6779313076649b7827999110e9689",
        "tuning_suite/pima_diabetes.json": "e06431579ff49cbe4abad940e45cf20a2448c640af9f612289bf45b523298292",
        "tuning_suite/student_por.json": "f06f61136a822a50065c730d9753b2ef25ef53016ff3ddf412650de5c44b806d",
        "tuning_suite/titanic.json": "c4b09e418a4963b4f291d15a4009493a2d3e1b40018e4f37b3b5e3912eeee855",
    },
}


def log(msg):
    print(msg, flush=True)


_HTTP_CACHE = {}   # archives shared by several files are fetched once


def http_get(url):
    if url in _HTTP_CACHE:
        return _HTTP_CACHE[url]
    _HTTP_CACHE[url] = _http_get(url)
    return _HTTP_CACHE[url]


def _http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read()


def fetch_bytes(fetch, repo_root):
    kind = fetch[0]
    if kind == "http":
        return http_get(fetch[1])
    if kind == "zip":
        _, url, member = fetch
        with zipfile.ZipFile(io.BytesIO(http_get(url))) as zf:
            return zf.read(member)
    if kind == "zipzip":
        _, url, outer_member, member = fetch
        with zipfile.ZipFile(io.BytesIO(http_get(url))) as zf:
            inner = zf.read(outer_member)
        with zipfile.ZipFile(io.BytesIO(inner)) as zf2:
            return zf2.read(member)
    raise ValueError("unknown fetch kind: %r" % (kind,))


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def check_sha256(data, expected, label, no_verify):
    got = sha256_hex(data)
    if got == expected:
        log("  OK    %s" % label)
        return True
    msg = ("sha256 MISMATCH for %s\n    expected %s\n    got      %s"
           % (label, expected, got))
    if no_verify:
        log("  WARN  %s (--no-verify, continuing)" % msg)
        return False
    sys.exit("ERROR: %s\n\nRe-run with --no-verify to proceed anyway (not recommended -- "
              "the converted JSON will then likely differ from the committed reference)."
              % msg)


def download_raw_files(suites, repo_root, no_verify):
    printed_move1_note = False
    ok = True
    for spec in RAW_FILES:
        if spec["suite"] not in suites:
            continue
        dest = os.path.join(repo_root, spec["dest"])
        if spec["fetch"][0] == "manual":
            if not os.path.exists(dest):
                if not printed_move1_note:
                    log(MOVE1_NOTE)
                    printed_move1_note = True
                log("Skipping %s (not present)" % spec["dest"])
                continue
            with open(dest, "rb") as f:
                ok = check_sha256(f.read(), spec["sha256"], spec["dest"], no_verify) and ok
            continue
        log("Fetching %s" % spec["dest"])
        data = fetch_bytes(spec["fetch"], repo_root)
        good = check_sha256(data, spec["sha256"], spec["dest"], no_verify)
        ok = ok and good
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
    return ok


def run_converter(repo_root, script_relpath, suite_dir=None, args=()):
    cmd = [sys.executable, os.path.join(repo_root, script_relpath)]
    if suite_dir:
        cmd.append(suite_dir)
    cmd.extend(args)
    log("Running %s ..." % script_relpath)
    subprocess.run(cmd, cwd=repo_root, check=True)


def verify_json(suites, repo_root, no_verify):
    all_ok = True
    for suite in suites:
        for relpath, expected in sorted(EXPECTED_JSON.get(suite, {}).items()):
            path = os.path.join(repo_root, relpath)
            if not os.path.exists(path):
                if relpath == "tuning_suite/move1.json":
                    continue      # optional, see MOVE1_NOTE
                log("  MISSING  %s" % relpath)
                all_ok = False
                continue
            with open(path, "rb") as f:
                data = f.read()
            good = check_sha256(data, expected, relpath, no_verify)
            all_ok = all_ok and good
    return all_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["test", "tuning", "text", "recent", "all"], default="all",
                     help="which suite to fetch and build (default: all)")
    ap.add_argument("--no-verify", action="store_true",
                     help="download/convert even if a sha256 check fails "
                          "(prints a warning instead of exiting)")
    args = ap.parse_args()

    suites = ["test", "tuning", "text", "recent"] if args.suite == "all" else [args.suite]

    log("=== Downloading raw sources ===")
    raw_ok = download_raw_files(suites, REPO_ROOT, args.no_verify)

    log("\n=== Running converters ===")
    if "test" in suites:
        run_converter(REPO_ROOT, "src/tools/build_test_suite.py")
    if "tuning" in suites:
        run_converter(REPO_ROOT, "src/tools/build_tuning_suite.py")
    if "text" in suites:
        run_converter(REPO_ROOT, "src/tools/build_text_suite.py")
    if "recent" in suites:
        run_converter(REPO_ROOT, "src/tools/build_recent_suite.py", args=["inflation_abstracts"])

    log("\n=== Verifying converted JSON ===")
    json_ok = verify_json(suites, REPO_ROOT, args.no_verify)

    if raw_ok and json_ok:
        log("\nAll files downloaded and verified byte-identical to the committed reference.")
    else:
        log("\nCompleted with mismatches (see WARN/MISMATCH lines above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
