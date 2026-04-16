"use client";
import React from 'react';

function Filters() {
  function handlePriceInput(e) {
    const rangeInput = document.querySelectorAll(".range-input input");
    const priceInput = document.querySelectorAll(".price-input input");
    const range = document.querySelector(".slider .progress");
    let priceGap = 1000;

    let minPrice = parseInt(priceInput[0].value);
    let maxPrice = parseInt(priceInput[1].value);

    if (maxPrice - minPrice >= priceGap && maxPrice <= rangeInput[1].max) {
      if (e.target.className === "input-min") {
        rangeInput[0].value = minPrice;
        range.style.left = (minPrice / rangeInput[0].max) * 100 + "%";
      } else {
        rangeInput[1].value = maxPrice;
        range.style.right = 100 - (maxPrice / rangeInput[1].max) * 100 + "%";
      }
    }
  }

  function handleRangeInput(e) {
    const rangeInput = document.querySelectorAll(".range-input input");
    const priceInput = document.querySelectorAll(".price-input input");
    const range = document.querySelector(".slider .progress");
    let priceGap = 1000;

    let minVal = parseInt(rangeInput[0].value);
    let maxVal = parseInt(rangeInput[1].value);

    if (maxVal - minVal < priceGap) {
      if (e.target.className === "range-min") {
        rangeInput[0].value = maxVal - priceGap;
      } else {
        rangeInput[1].value = minVal + priceGap;
      }
    } else {
      priceInput[0].value = minVal;
      priceInput[1].value = maxVal;
      range.style.left = (minVal / rangeInput[0].max) * 100 + "%";
      range.style.right = 100 - (maxVal / rangeInput[1].max) * 100 + "%";
    }
  }

  return (
    <div className="col-lg-4 order-first order-lg-0">
      <div className="aside mb-4 mb-lg-0">
        <div className="search-form">
          <input type="text" placeholder="Search here..." className="form-control" />
          <span className="icon"> <i className="far fa-search"></i> </span>
        </div>
        <h6 className="sub-title fw-bold fsz-16 text-capitalize mb-30 mt-60"> Filter By Pice </h6>
        <div className="price-card">
          <div className="slider">
            <div className="progress"></div>
          </div>
          <div className="range-input">
            <input type="range" className="range-min" min="0" max="10000" defaultValue="0" step="100" onInput={handleRangeInput} />
            <input type="range" className="range-max" min="0" max="10000" defaultValue="10000" step="100" onInput={handleRangeInput} />
          </div>
          <div className="row mt-20 align-items-center">
            <div className="col-6">
              <div className="price-input">
                <p className="color-777"> Price: </p>
                <div className="field">
                  <input type="number" className="input-min" defaultValue="0" onInput={handlePriceInput} />
                </div>
                <div className="separator"></div>
                <div className="field">
                  <input type="number" className="input-max" defaultValue="10000" onInput={handlePriceInput} />
                </div>
              </div>
            </div>
            <div className="col-6 text-end">
              <strong className="color-blue1 ms-auto text-capitalize"> dollar </strong>
            </div>
          </div>
        </div>

        <h6 className="sub-title fw-bold fsz-16 text-capitalize mb-30 mt-60"> Category </h6>
        <ul className="cat-links">
          <li> <a href="#"> Design </a> <span className="fal fa-long-arrow-right"></span> </li>
          <li> <a href="#"> Frames </a> <span className="fal fa-long-arrow-right"></span> </li>
          <li> <a href="#"> Items </a> <span className="fal fa-long-arrow-right"></span> </li>
          <li> <a href="#"> T-shirts </a> <span className="fal fa-long-arrow-right"></span> </li>
        </ul>

        <h6 className="sub-title fw-bold fsz-16 text-capitalize mb-30 mt-60"> Keywords </h6>
        <ul className="cat-links">
          <li> <a href="#"> Innovative </a> <span className="fal fa-long-arrow-right"></span> </li>
          <li> <a href="#"> Print </a> <span className="fal fa-long-arrow-right"></span> </li>
          <li> <a href="#"> Project </a> <span className="fal fa-long-arrow-right"></span> </li>
          <li> <a href="#"> Teamwork </a> <span className="fal fa-long-arrow-right"></span> </li>
          <li> <a href="#"> Technology </a> <span className="fal fa-long-arrow-right"></span> </li>
        </ul>

      </div>
    </div>
  )
}

export default Filters