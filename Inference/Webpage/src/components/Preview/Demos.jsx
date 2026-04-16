import React from 'react';
import Link from 'next/link';
//= Static Data
import data from '@/data/Preview/demos.json';

function Demos() {
  return (
    <section className="demos section-padding pt-80">
      <div className="container-xxl">
        <div className="row">
          {data.map(item => (
            <div className="col-lg-4 col-md-6" key={item.id}>
              <div className={`item text-center ${item.isDark ? 'dark_item' : ''}`}>
                <Link href={item.isDark ? item.link + '?mode=dark' : item.link} target="_blank">
                  <div className="img">
                    <img src={item.image} alt="" />
                  </div>
                  <h6 className="mt-15">{item.title} <span>{item.isDark && '(dark)'}</span></h6>
                </Link>
              </div>
            </div>
          ))}
          <div className="col-lg-4 col-md-6">
            <div className="item text-center">
              <div className="img">
                <img src="/landing-preview/img/demos/more.jpg" alt="" />
              </div>
              <h6 className="mt-15">Coming More</h6>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Demos