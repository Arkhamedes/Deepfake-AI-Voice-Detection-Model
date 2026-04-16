"use client";
import React, { useEffect } from 'react';
import a from 'next/link';

function Navbar() {
  useEffect(() => {
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  function handleScroll() {
    const bodyScroll = window.scrollY;
    const navbar = document.querySelector(".navbar");

    if (bodyScroll > 300) navbar.classList.add("nav-scroll");
    else navbar.classList.remove("nav-scroll");
  }

  function handleDropdownMouseMove(event) {
    event.currentTarget.querySelector('.dropdown-menu').classList.add('show');
  }

  function handleDropdownMouseLeave(event) {
    event.currentTarget.querySelector('.dropdown-menu').classList.remove('show');
  }

  function toggleNavbar() {
    document.querySelector(".navbar .navbar-collapse").classList.toggle("show");
  }

  return (
    <nav className="navbar navbar-expand-lg static">
      <div className="container">

        <a className="logo icon-img-100" href="#">
          <img src="/landing-preview/img/logo.png" alt="logo" className="lt_item" />
          <img src="/landing-preview/img/logo_wh.png" alt="logo" className="dr_item" />
        </a>

        <button className="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarSupportedContent"
          aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
          <span className="icon-bar"><i className="fas fa-bars"></i></span>
        </button>

        <div className="collapse navbar-collapse justify-content-center" id="navbarSupportedContent">
          <ul className="navbar-nav">
            <li className="nav-item dropdown" onMouseMove={handleDropdownMouseMove} onMouseLeave={handleDropdownMouseLeave}>
              <a className="nav-link dropdown-toggle" data-toggle="dropdown" href="#" role="button"
                aria-haspopup="true" aria-expanded="false"><span className="rolling-text">Home</span></a>
              <div className="dropdown-menu mega-menu">
                <div className="container">
                  <div className="row">
                    <div className="col-lg-3">
                      <div className="clumn">
                        <div className="title">
                          <h6 className="sub-title ls1">Home Light</h6>
                        </div>
                        <div className="links">
                          <a className="dropdown-item" href="/home-1"> Agency Portfolio </a>
                          <a className="dropdown-item" href="/home-2"> Creative Studio </a>
                          <a className="dropdown-item" href="/home-3"> Modern Startup </a>
                          <a className="dropdown-item" href="/home-4"> Apps Technology </a>
                          <a className="dropdown-item" href="/home-5"> Creative Agency </a>
                          <a className="dropdown-item" href="/home-6"> Business Showcase </a>
                          <a className="dropdown-item" href="/home-7"> Consulting Agency </a>
                          <a className="dropdown-item" href="/home-8"> Personal Portfolio </a>
                          <a className="dropdown-item" href="/home-9"> Shop Home </a>
                          <a className="dropdown-item" href="/home-10"> Tech Landing </a>

                        </div>
                      </div>
                    </div>
                    <div className="col-lg-3">
                      <div className="clumn">
                        <div className="title">
                          <h6 className="sub-title ls1">Home Dark</h6>
                        </div>
                        <div className="links">
                          <a className="dropdown-item" href="/home-1?mode=dark"> Agency Portfolio </a>
                          <a className="dropdown-item" href="/home-2?mode=dark"> Creative Studio </a>
                          <a className="dropdown-item" href="/home-3?mode=dark"> Modern Startup </a>
                          <a className="dropdown-item" href="/home-4?mode=dark"> Apps Technology </a>
                          <a className="dropdown-item" href="/home-5?mode=dark"> Creative Agency </a>
                          <a className="dropdown-item" href="/home-6?mode=dark"> Business Showcase </a>
                          <a className="dropdown-item" href="/home-7?mode=dark"> Consulting Agency </a>
                          <a className="dropdown-item" href="/home-8?mode=dark"> Personal Portfolio </a>
                          <a className="dropdown-item" href="/home-9?mode=dark"> Shop Home </a>
                          <a className="dropdown-item" href="/home-10?mode=dark"> Tech Landing </a>
                        </div>
                      </div>
                    </div>
                    <div className="col-lg-3">
                      <div className="clumn">
                        <div className="title">
                          <h6 className="sub-title ls1">Portfolio</h6>
                        </div>
                        <div className="links">
                          <a className="dropdown-item" href="/inner_pages/page_portfolio_grid"> Portfolio grid </a>
                          <a className="dropdown-item" href="/inner_pages/page_portfolio_masonry"> Portfolio masonry </a>
                          <a className="dropdown-item" href="/inner_pages/page_portfolio_showcase"> Portfolio showcase </a>
                          <a className="dropdown-item" href="/inner_pages/page_gallery"> Portfolio gallery </a>
                          <a className="dropdown-item" href="/inner_pages/page_single_project"> Project Details </a>
                        </div>
                      </div>
                    </div>
                    <div className="col-lg-3">
                      <div className="clumn">
                        <div className="title">
                          <h6 className="sub-title ls1">E-commerce</h6>
                        </div>
                        <div className="links">
                          <a className="dropdown-item" href="/inner_pages/page_product_grid"> Shop List </a>
                          <a className="dropdown-item" href="/inner_pages/page_product_grid2"> Shop grid </a>
                          <a className="dropdown-item" href="/inner_pages/page_wishlist"> Wishlist </a>
                          <a className="dropdown-item" href="/inner_pages/page_product_single"> Single Product </a>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </li>
            <li className="nav-item dropdown" onMouseMove={handleDropdownMouseMove} onMouseLeave={handleDropdownMouseLeave}>
              <a className="nav-link dropdown-toggle" data-toggle="dropdown" href="#" role="button"
                aria-haspopup="true" aria-expanded="false"><span className="rolling-text">Pages</span></a>
              <div className="dropdown-menu">
                <a className="dropdown-item" href="/inner_pages/page_about_us">About</a>
                <a className="dropdown-item" href="/inner_pages/page_services">Services</a>
                <a className="dropdown-item" href="/inner_pages/page_service_det">Service detail</a>
                <a className="dropdown-item" href="/inner_pages/page_team">Our Team</a>
                <a className="dropdown-item" href="/inner_pages/page_team_member">Team member</a>
                <a className="dropdown-item" href="/inner_pages/page_faq">FAQ</a>
                <a className="dropdown-item" href="/inner_pages/page_404">404</a>
                <a className="dropdown-item" href="/inner_pages/page_contact">Contact Us</a>
                <a className="dropdown-item" href="/inner_pages/page_soon">More soon</a>
              </div>
            </li>
            <li className="nav-item dropdown" onMouseMove={handleDropdownMouseMove} onMouseLeave={handleDropdownMouseLeave}>
              <a className="nav-link dropdown-toggle" data-toggle="dropdown" href="#" role="button"
                aria-haspopup="true" aria-expanded="false"><span className="rolling-text">Portfolio</span></a>
              <ul className="dropdown-menu">
                <li><a className="dropdown-item" href="/inner_pages/page_portfolio_grid">Portfolio grid</a></li>
                <li><a className="dropdown-item" href="/inner_pages/page_portfolio_masonry">Portfolio masonry</a></li>
                <li><a className="dropdown-item" href="/inner_pages/page_portfolio_showcase">Portfolio showcase</a></li>
                <li><a className="dropdown-item" href="/inner_pages/page_gallery">Portfolio gallery</a></li>
                <li><a className="dropdown-item" href="/inner_pages/page_single_project">Project Details</a></li>
              </ul>
            </li>
            <li className="nav-item dropdown" onMouseMove={handleDropdownMouseMove} onMouseLeave={handleDropdownMouseLeave}>
              <a className="nav-link dropdown-toggle" data-toggle="dropdown" href="#" role="button"
                aria-haspopup="true" aria-expanded="false"><span className="rolling-text">Blogs</span></a>
              <div className="dropdown-menu">
                <a className="dropdown-item" href="/inner_pages/page_blog_list">Blog list</a>
                <a className="dropdown-item" href="/inner_pages/page_blog_grid">Blog grid</a>
                <a className="dropdown-item" href="/inner_pages/page_blog_masonry">Blog masonry</a>
                <a className="dropdown-item" href="/inner_pages/page_blog_details">Single post</a>
                <a className="dropdown-item" href="/inner_pages/page_blog_details_overlay">Overlay post</a>
              </div>
            </li>
            <li className="nav-item dropdown" onMouseMove={handleDropdownMouseMove} onMouseLeave={handleDropdownMouseLeave}>
              <a className="nav-link dropdown-toggle" data-toggle="dropdown" href="#" role="button"
                aria-haspopup="true" aria-expanded="false"><span className="rolling-text">Shop</span></a>
              <div className="dropdown-menu">
                <a className="dropdown-item" href="/inner_pages/page_product_grid">Shop List</a>
                <a className="dropdown-item" href="/inner_pages/page_product_grid2">Shop grid</a>
                <a className="dropdown-item" href="/inner_pages/page_wishlist">Wishlist</a>
                <a className="dropdown-item" href="/inner_pages/page_product_single">Single Product</a>
              </div>
            </li>
            <li className="nav-item">
              <a className="nav-link" href="/inner_pages/page_contact"><span className="rolling-text">Contact</span></a>
            </li>
          </ul>
        </div>

        <div className="purchase">
          <a href="#0" className="butn butn-md radius-30 bg-dark">
            <span>Purchase</span>
          </a>
        </div>
      </div>
    </nav>
  )
}

export default Navbar