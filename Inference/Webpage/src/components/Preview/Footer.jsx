import React from 'react'

function Footer() {
  return (
    <footer>
      <div className="sub-footer pt-40 pb-40 ontop sub-bg">
        <div className="container">
          <div className="row">
            <div className="col-lg-4">
              <div className="logo">
                <a href="#0">
                  <img src="/landing-preview/img/logo.png" alt="logo" className="lt_item" />
                  <img src="/landing-preview/img/logo_wh.png" alt="logo" className="dr_item" />
                </a>
              </div>
            </div>
            <div className="col-lg-8">
              <div className="copyright d-flex">
                <div className="ml-auto">
                  <p className="fz-13">© 2023 Swak is Proudly Powered by <span className="underline"><a href="#" target="_blank">ThemesCamp</a></span></p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default Footer