import React from 'react';
//= Components
import Filters from './Filters';
//= Static Data
import data from '@/data/InnerPages/Product/grid2.json';

function Grid2() {
  return (
    <section className="tc-product-grid-style1">
      <div className="container">
        <div className="row gx-0">
          <div className="col-lg-8">
            <div className="products-side">
              <div className="top-filter">
                <div className="row align-items-center">
                  <div className="col-lg-6">
                    <h6 className="prod-count color-666 fsz-16 mb-0"> Showing <strong className="color-000"> 1-12 </strong> of <strong className="color-000"> 18 </strong> results </h6>
                  </div>
                  <div className="col-lg-6 mb-4 mb-lg-0 order-first order-lg-0">
                    <div className="d-flex justify-content-lg-end">
                      <select name="" id="" className="form-select border-0 w-auto fw-bold ps-0">
                        <option value="1"> Sort By Latest </option>
                        <option value="1"> Sort By Best </option>
                      </select>
                    </div>
                  </div>
                </div>
              </div>
              <div className="products">
                <div className="row">
                  {data.map(item => (
                    <div className="col-lg-6" key={item.id}>
                      <div className="product-card mt-60 wow fadeInUp slow">
                        <div className="img th-400 img-cover">
                          <img src={item.image} alt="" />
                        </div>
                        <div className="info pt-30">
                          <a href="#" className="fsz-16 color-666"> {item.name} </a>
                          <h6 className="fsz-20 fw-bold mt-2 mb-0"> {item.price} </h6>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
          {/* Filters */}
          <Filters />
          {/* Paginations */}
          <div className="col-lg-8">
            <ul className="pagination">
              <li className="page-item">
                <a className="page-link" href="#" aria-label="Previous">
                  <span aria-hidden="true">«</span>
                </a>
              </li>
              <li className="page-item"><a className="page-link" href="#">1</a></li>
              <li className="page-item"><a className="page-link" href="#">2</a></li>
              <li className="page-item"><a className="page-link" href="#">3</a></li>
              <li className="page-item"><a className="page-link" href="#">...</a></li>
              <li className="page-item"><a className="page-link" href="#">10</a></li>
              <li className="page-item">
                <a className="page-link" href="#" aria-label="Next">
                  <span aria-hidden="true">»</span>
                </a>
              </li>
            </ul>
          </div>

        </div>

      </div>
    </section>
  )
}

export default Grid2