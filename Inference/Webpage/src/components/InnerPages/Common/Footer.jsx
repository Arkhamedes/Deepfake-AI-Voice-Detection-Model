import React from 'react';
import Link from 'next/link';

function Footer() {
  return (
    <footer className="tc-inner-footer-style1">
      <div className="container">
        <div className="foot-content">
          <div className="info text-center">
            <p className="fsz-30 mb-10 text-uppercase js-splittext-lines"> got a project in mind? </p>
            <h2 className="mb-80 js-splittext-lines"> <Link href="/inner_pages/page_contact"> let’s talk! </Link> </h2>
            <p> <a href="#" className="fsz-16 color-999 text-uppercase hover-blue1 lh-6 js-splittext-lines"> hello@Swakstudio.co </a> </p>
            <p> <a href="#" className="fsz-16 color-999 text-uppercase hover-blue1 lh-6 js-splittext-lines"> 90 Fairground rd, florida 3290, us </a> </p>
            <p> <a href="#" className="fsz-16 color-999 text-uppercase hover-blue1 lh-6 js-splittext-lines"> +32 (0)50 31 28 32 </a> </p>
            <div className="social-icons js-splittext-lines">
              <a href="#"> <i className="lab la-twitter"></i> </a>
              <a href="#"> <i className="lab la-facebook-f"></i> </a>
              <a href="#"> <i className="lab la-instagram"></i> </a>
              <a href="#"> <i className="lab la-youtube"></i> </a>
            </div>
          </div>
        </div>
        <div className="foot">
          <div className="row">
            <div className="col-lg-3 text-center text-lg-start">
              <p className="fsz-14 color-999 text-uppercase"> © 2023 By <a href="#" className="text-white"> Swak studio </a> </p>
            </div>
            <div className="col-lg-6">
              <div className="links my-3 my-lg-0">
                <Link href="/inner_pages/page_services"> services </Link>
                <Link href="/inner_pages/page_portfolio_masonry"> works </Link>
                <Link href="/inner_pages/page_about_us"> about </Link>
                <Link href="/inner_pages/page_blog_list"> news </Link>
                <Link href="/inner_pages/page_contact"> contact </Link>
                <a href="#"> policy </a>
              </div>
            </div>
            <div className="col-lg-3 text-lg-end d-none d-lg-block">
              <div className="dropdown">
                <button className="border-0 bg-transparent p-0 text-white dropdown-toggle" type="button" id="dropdownMenuButton1" data-bs-toggle="dropdown" aria-expanded="false">
                  <i className="ti-world me-1"></i> ENG
                </button>
                <ul className="dropdown-menu" aria-labelledby="dropdownMenuButton1">
                  <li><a className="dropdown-item" href="#">FR</a></li>
                  <li><a className="dropdown-item" href="#">AR</a></li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default Footer