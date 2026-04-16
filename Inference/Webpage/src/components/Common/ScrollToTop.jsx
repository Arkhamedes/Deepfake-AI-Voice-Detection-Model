"use client";
import React, { useEffect } from 'react';

function ScrollToTop() {
  useEffect(() => {
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  function handleScroll() {
    let toTop = document.querySelector(".to_top")
    if (window.scrollY > 700) toTop.classList.add("show");
    else toTop.classList.remove("show");
  }

  function scrollToTop(event) {
    event.preventDefault();
    window.scrollTo(0, 0)
  }

  return (
    <a href="#" className="to_top" onClick={scrollToTop}>
      <i className="la la-angle-up"></i>
    </a>
  )
}

export default ScrollToTop