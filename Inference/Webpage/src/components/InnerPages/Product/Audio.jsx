"use client";
import React, { useState } from "react";

function Single() {
  const [audio, setAudio] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [result, setResult] = useState(null); // ✅ NEW
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setAudio(URL.createObjectURL(file));
      setAudioFile(file);
      setResult(null); // reset old result
    }
  };

  const handleClick = async () => {
    try {
      if (!audioFile) {
        alert("Please upload an MP3 file first");
        return;
      }

      setLoading(true);

      const formData = new FormData();
      formData.append("file", audioFile);

      const res = await fetch("http://localhost:8000/predict-audio", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      console.log(data);

      setResult(data); // ✅ SAVE JSON RESULT

    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // ✅ Safe derived values
  const isFake = result?.final_prediction === 1 || result?.final_prediction === true;
  const confidence = result?.confidence ?? 0;

  return (
    <section className="tc-product-single-style1">
      <div className="main-info mb-100">
        <div className="container">
          <div className="row gx-0">

            {/* LEFT - AUDIO */}
            <div className="col-lg-5">
              <div className="product-img mb-4 mb-lg-0">
                <div onClick={() => document.getElementById("fileInput").click()}>
                  <input
                    id="fileInput"
                    type="file"
                    accept="audio/mpeg"
                    style={{ display: "none" }}
                    onChange={handleChange}
                  />

                  {audio ? (
                    <>
                      <p><strong>Uploaded Audio</strong></p>
                      <audio controls style={{ width: "100%" }}>
                        <source src={audio} type="audio/mpeg" />
                      </audio>
                    </>
                  ) : (
                    <div style={{
                      width: "100%",
                      height: "200px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      border: "2px dashed #ccc",
                      borderRadius: "8px",
                      cursor: "pointer"
                    }}>
                      Click to upload MP3
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* MIDDLE - BUTTON */}
            <div className="col-lg-3">
              <div className="product-info">
                <h3 className="fsz-26 fw-bold mb-30">
                  Audio Classifier
                </h3>

                <button
                  onClick={handleClick}
                  disabled={loading}
                  style={{
                    padding: "12px 24px",
                    background: "#007bff",
                    color: "#fff",
                    border: "none",
                    borderRadius: "6px",
                    cursor: "pointer"
                  }}
                >
                  {loading ? "Processing..." : "Run Detection"}
                </button>
              </div>
            </div>

            {/* RIGHT - RESULT PANEL */}
            <div className="col-lg-4">
              <div style={{
                padding: "20px",
                border: "1px solid #eee",
                borderRadius: "10px",
                marginTop: "20px"
              }}>

                <h4>Detection Result</h4>

                {loading ? (
                  <p>Analyzing audio...</p>
                ) : !result ? (
                  <p>No result yet</p>
                ) : (
                  <>
                    {/* FINAL RESULT */}
                    <h2 style={{ color: isFake ? "red" : "green" }}>
                      {isFake ? "FAKE / AI" : "REAL"}
                    </h2>

                    {/* CONFIDENCE */}
                    <p>
                      Confidence: {(confidence * 100).toFixed(1)}%
                    </p>

                    {/* BAR */}
                    <div style={{
                      width: "100%",
                      height: "20px",
                      background: "#eee",
                      borderRadius: "10px",
                      overflow: "hidden",
                      marginBottom: "15px"
                    }}>
                      <div style={{
                        width: `${confidence * 100}%`,
                        height: "100%",
                        background: isFake ? "red" : "green",
                        transition: "width 0.4s ease"
                      }} />
                    </div>

                    {/* SEGMENT INFO */}
                    <p><strong>Segments:</strong> {result.num_segments}</p>

                    {/* SEGMENT PREDICTIONS */}
                    <div style={{
                      maxHeight: "150px",
                      overflowY: "auto",
                      fontSize: "14px",
                      background: "#f9f9f9",
                      padding: "10px",
                      borderRadius: "6px"
                    }}>
                      {result.segment_predictions.map((p, i) => (
                        <div key={i}>
                          Segment {i + 1}: {p}
                        </div>
                      ))}
                    </div>

                  </>
                )}

              </div>
            </div>

          </div>
        </div>
      </div>
    </section>
  );
}

export default Single;