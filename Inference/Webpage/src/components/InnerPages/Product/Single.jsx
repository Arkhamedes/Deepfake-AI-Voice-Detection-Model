"use client";
import React, { useState } from "react";
import { useRouter } from "next/navigation";

function Single() {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file); // ✅ real file
      setImagePreview(URL.createObjectURL(file)); // ✅ preview only
      
    }
  };

  const handleClick = async () => {
  try {
    if (!image) {
      alert("Please upload an image first");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("file", image);

    const res = await fetch("http://localhost:8000/predict-image", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const text = await res.text();
      console.error(text);
      throw new Error("API failed");
    }

    const blob = await res.blob();
    const newImageUrl = URL.createObjectURL(blob);

    setImagePreview(newImageUrl);

  } catch (err) {
    console.error(err);
  } finally {
    setLoading(false);
  }
};

  return (
    <section className="tc-product-single-style1">
      <div className="main-info mb-100">
        <div className="container">
          <div className="row gx-0">
            
            {/* Image Upload */}
            <div className="col-lg-5">
              <div className="product-img mb-4 mb-lg-0">
                <div onClick={() => document.getElementById("fileInput").click()}>
                  <input
                    id="fileInput"
                    type="file"
                    accept="image/*"
                    style={{ display: "none" }}
                    onChange={handleChange}
                  />

                  <img
                    src={imagePreview || "/inner_pages/assets/img/products/upload.png"}
                    alt=""
                    style={{
                      cursor: "pointer",
                      width: "640px",
                      height: "640px",
                      objectFit: "cover",
                      borderRadius: "8px"
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Info + Button */}
            <div className="col-lg-6 offset-lg-1">
              <div className="product-info">
                
                <h3 className="fsz-26 fw-bold text-capitalize mb-30">
                  Computer Vision to detect PCB faults
                </h3>

                <div className="text fsz-16 color-777 mb-30">
                  Lorem ipsum dolor sit amet, consectetur adipiscing elit...
                </div>

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

          </div>
        </div>
      </div>
    </section>
  );
}

export default Single;