import React from 'react';

function ProductInfo() {
  return (
    <div className="sub-info">
      <div className="container">
        <div className="row">
          <div className="col-lg-3">
            <ul className="nav nav-pills d-block" id="pills-tab" role="tablist">
              <li className="nav-item" role="presentation">
                <button className="nav-link active" id="pills-tab1-tab" data-bs-toggle="pill" data-bs-target="#pills-tab1" type="button" role="tab" aria-controls="pills-tab1" aria-selected="true">Description</button>
              </li>
              <li className="nav-item" role="presentation">
                <button className="nav-link" id="pills-tab2-tab" data-bs-toggle="pill" data-bs-target="#pills-tab2" type="button" role="tab" aria-controls="pills-tab2" aria-selected="false">Additional Information</button>
              </li>
              <li className="nav-item" role="presentation">
                <button className="nav-link" id="pills-tab3-tab" data-bs-toggle="pill" data-bs-target="#pills-tab1" type="button" role="tab" aria-controls="pills-tab3" aria-selected="false">Reviews <span> (2) </span></button>
              </li>
              <li className="nav-item" role="presentation">
                <button className="nav-link" id="pills-tab4-tab" data-bs-toggle="pill" data-bs-target="#pills-tab2" type="button" role="tab" aria-controls="pills-tab4" aria-selected="false">Size Chart</button>
              </li>
            </ul>
          </div>
          <div className="col-lg-9">
            <div className="tab-content" id="pills-tabContent">
              <div className="tab-pane fade show active" id="pills-tab1" role="tabpanel" aria-labelledby="pills-tab1-tab" tabIndex="0">
                <p className="text fsz-16 color-777 mb-20">
                  In marketing a product is an object or system made available for consumer use it is anything that can be offered to a market to the desire or need of aretailing, products are often referred to as merchandise, and in manufacturing, products are bought as materials and then sold as finished goods. A service regarded to as a type of product. Commodities are usually raw materials metals and agricultural products, but a commodity can also be anything wide the open market. In project management, the formal definition of the project deliverables
                </p>
                <p className="text fsz-16 color-777">
                  A product can be classNameified as tangible or intangible. A tangible product is a physical object that can be perceived by touch  building, vehicle, gadget, An intangible product is a product that can only be perceived indirectly such as an insurance policy.  can be broadly classNameified under intangible be durable or non durable. A product line is "a group of products that are closely either because they function in a similar manner, are sold to the same customergroups.
                </p>
              </div>
              <div className="tab-pane fade" id="pills-tab2" role="tabpanel" aria-labelledby="pills-tab2-tab" tabIndex="0">
                <div className="tab-pane fade show active" id="pills-tab1" role="tabpanel" aria-labelledby="pills-tab1-tab" tabIndex="0">
                  <p className="text fsz-16 color-777 mb-20">
                    In marketing a product is an object or system made available for consumer use it is anything that can be offered to a market to the desire or need of aretailing, products are often referred to as merchandise, and in manufacturing, products are bought as materials and then sold as finished goods. A service regarded to as a type of product. Commodities are usually raw materials metals and agricultural products, but a commodity can also be anything wide the open market. In project management, the formal definition of the project deliverables
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ProductInfo