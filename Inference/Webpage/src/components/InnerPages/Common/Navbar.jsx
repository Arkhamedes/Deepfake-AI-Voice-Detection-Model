import React from 'react';
import Link from 'next/link';

function Navbar({ darkLogo }) {
  return (
    <nav className="navbar navbar-expand-lg navbar-dark tc-inner-navbar-style1">
      <div className="container-fluid">
        <a className="navbar-brand" href="#">
          <img src={`/inner_pages/assets/img/logo_home2${darkLogo ? '_drk' : ''}.png`} alt="" className="logo" />
        </a>
        <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent"
          aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
          <span className="navbar-toggler-icon"></span>
        </button>
        <div className="collapse navbar-collapse mt-3 mt-lg-0" id="navbarSupportedContent">
          <ul className="navbar-nav mx-auto mb-2 mb-lg-0">
            <li className="nav-item">
              <Link className="nav-link active" aria-current="page" href="/inner_pages/page_services">services</Link>
            </li>
            <li className="nav-item">
              <Link className="nav-link" href="/inner_pages/page_portfolio_showcase">works</Link>
            </li>
            <li className="nav-item">
              <Link className="nav-link" href="/inner_pages/page_about_us">about</Link>
            </li>
            <li className="nav-item">
              <Link className="nav-link" href="/inner_pages/page_blog_list">news</Link>
            </li>
            <li className="nav-item">
              <Link className="nav-link" href="/inner_pages/page_contact">contact</Link>
            </li>
          </ul>
          <div className="dropdown">
            <button className={`border-0 bg-transparent p-0 ${darkLogo ? '' : 'text-white'} dropdown-toggle`} type="button" id="dropdownMenuButton1" data-bs-toggle="dropdown"
              aria-expanded="false">
              <i className="ti-world me-1"></i> ENG
            </button>
            <ul className="dropdown-menu" aria-labelledby="dropdownMenuButton1">
              <li><a className="dropdown-item" href="#">FR</a></li>
              <li><a className="dropdown-item" href="#">AR</a></li>
            </ul>
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar