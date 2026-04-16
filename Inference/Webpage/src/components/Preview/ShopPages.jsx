import React from 'react';

function ShopPages() {
  return (
    <section className="shop-demos section-padding sub-bg">
      <div className="container">
        <div className="sec-lg-head text-center mb-50">
          <h2 className="fz-50 d-rotate wow">
            <span className="rotate-text">Carefully crafted.</span>
            <span className="rotate-text">Woocommerce Demos.</span>
          </h2>
        </div>
      </div>
      <div className="container-fluid">
        <div className="row">
          <div className="col-lg-3 col-md-6">
            <div className="item">
              <div className="img">
                <img src="/landing-preview/img/shop/1.jpg" alt="" />
              </div>
              <h6>Main Shop</h6>
              <a href="#0" target="_blank"></a>
            </div>
          </div>
          <div className="col-lg-3 col-md-6">
            <div className="item">
              <div className="img">
                <img src="/landing-preview/img/shop/01.jpg" alt="" />
              </div>
              <h6>Filter Shop</h6>
              <a href="#0" target="_blank"></a>
            </div>
          </div>
          <div className="col-lg-3 col-md-6">
            <div className="item">
              <div className="img">
                <img src="/landing-preview/img/shop/2.jpg" alt="" />
              </div>
              <h6>Wishlist</h6>
              <a href="#0" target="_blank"></a>
            </div>
          </div>
          <div className="col-lg-3 col-md-6">
            <div className="item">
              <div className="img">
                <img src="/landing-preview/img/shop/02.jpg" alt="" />
              </div>
              <h6>Single Product</h6>
              <a href="#0" target="_blank"></a>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default ShopPages