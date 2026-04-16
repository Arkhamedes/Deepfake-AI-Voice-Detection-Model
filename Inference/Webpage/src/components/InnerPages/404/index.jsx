import React from 'react'

function NotFound() {
  return (
    <header className="tc-404-header-style1">
      <div className="logo">
        <img src="/inner_pages/assets/img/logo_home2.png" alt="" />
      </div>
      <div className="container">
        <div className="header-content">
          <h1> 404 </h1>
          <h2 className="fsz-50"> Oops! It looks like the page you were trying <br /> to access is not available </h2>
          <div className="button_su border-0 mt-60">
            <span className="su_button_circle bg-000 desplode-circle"></span>
            <a href="#" className="butn text-uppercase border-1 border_light border button_su_inner bg-transparent py-3 px-5">
              <span className="button_text_container fsz-14 text-white"> <i className="fal fa-long-arrow-left me-2"></i> Go back home </span>
            </a>
          </div>
        </div>
      </div>
      <img src="/inner_pages/assets/img/symbol_wh.png" alt="" className="symbol1" />
      <img src="/inner_pages/assets/img/symbol_wh.png" alt="" className="symbol2" />
    </header>
  )
}

export default NotFound