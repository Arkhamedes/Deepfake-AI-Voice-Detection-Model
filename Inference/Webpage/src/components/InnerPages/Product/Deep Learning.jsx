"use client";
import React, { useState } from "react";

function Single() {
  const [text, setText] = useState("");
  const [followers, setFollowers] = useState("");
  const [retweets, setRetweets] = useState("");
  const [favourites, setFavourites] = useState("");
  const [likes, setLikes] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleClick = async () => {
    try {
      if (!text) {
        alert("Please enter tweet text");
        return;
      }

      setLoading(true);

      const meta = [
        Number(followers) || 0,
        Number(retweets) || 0,
        Number(favourites) || 0,
        Number(likes) || 0
      ];

      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text,
          meta_features: meta,
        }),
      });

      const data = await res.json();
      console.log(data);

      setResult(data);

    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // ✅ Safe derived values
  const isBot = result?.prediction === true;
  const botProb = result?.confidence ?? 0;
  const humanProb = 1 - botProb;

  return (
    <section className="tc-product-single-style1">
      <div className="main-info mb-100">
        <div className="container">
          <div className="row gx-0">

            {/* LEFT SIDE */}
            <div className="col-lg-6 offset-lg-1">
              <div className="product-info">

                <h3 className="fsz-26 fw-bold text-capitalize mb-30">
                  Deep Learning Bot Tweet Classifier
                </h3>

                <label>Tweet</label>
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Enter your tweet here..."
                  rows={4}
                  style={{
                    display: "block",
                    width: "100%",
                    minHeight: "120px",
                    resize: "vertical",
                    padding: "10px",
                    marginBottom: "15px",
                    borderRadius: "6px",
                    border: "1px solid #ccc"
                  }}
                />

                <label>Followers</label>
                <input
                  type="number"
                  value={followers}
                  onChange={(e) => setFollowers(e.target.value)}
                  placeholder="Number of followers"
                  style={{ display: "block", width: "100%", marginBottom: "15px" }}
                />

                <label>Retweets</label>
                <input
                  type="number"
                  value={retweets}
                  onChange={(e) => setRetweets(e.target.value)}
                  placeholder="Number of retweets"
                  style={{ display: "block", width: "100%", marginBottom: "15px" }}
                />

                <label>Favourites</label>
                <input
                  type="number"
                  value={favourites}
                  onChange={(e) => setFavourites(e.target.value)}
                  placeholder="Number of favourites"
                  style={{ display: "block", width: "100%", marginBottom: "15px" }}
                />

                <label>Likes</label>
                <input
                  type="number"
                  value={likes}
                  onChange={(e) => setLikes(e.target.value)}
                  placeholder="Number of likes"
                  style={{ display: "block", width: "100%", marginBottom: "20px" }}
                />

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

            {/* RIGHT SIDE */}
            <div className="col-lg-4">
              <div style={{
                padding: "20px",
                border: "1px solid #eee",
                borderRadius: "10px",
                marginTop: "60px"
              }}>

                <h4>Detection Result</h4>

                {loading ? (
                  <p>Analyzing...</p>
                ) : !result ? (
                  <p>No result yet</p>
                ) : (
                  <>
                    {/* LABEL */}
                    <h2 style={{ color: isBot ? "red" : "green" }}>
                      {isBot ? "BOT" : "HUMAN"}
                    </h2>

                    {/* PROBABILITIES */}
                    <p>
                      Bot: {(botProb * 100).toFixed(1)}% | Human: {(humanProb * 100).toFixed(1)}%
                    </p>

                    {/* SPLIT BAR */}
                    <div style={{
                      width: "100%",
                      height: "20px",
                      background: "#eee",
                      borderRadius: "10px",
                      overflow: "hidden",
                      display: "flex"
                    }}>
                      <div style={{
                        width: `${botProb * 100}%`,
                        background: "red",
                        transition: "width 0.4s ease"
                      }} />
                      <div style={{
                        width: `${humanProb * 100}%`,
                        background: "green",
                        transition: "width 0.4s ease"
                      }} />
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