"use client";
import Link from 'next/link';
//= Modules
import { Swiper, SwiperSlide } from 'swiper/react';
import { Autoplay, Navigation, Pagination } from 'swiper';
//= Static Data
import data from '@/data/Preview/pages.json';

const swiperOptions = {
  modules: [Autoplay, Navigation, Pagination],
  slidesPerView: 3,
  spaceBetween: 60,
  loop: true,
  autoplay: true,
  speed: 1000,
  pagination: {
    el: '.swiper-pagination',
    clickable: true,
  },
  navigation: {
    nextEl: '.testim-controls .swiper-button-next',
    prevEl: '.testim-controls .swiper-button-prev'
  },
  breakpoints: {
    0: {
      slidesPerView: 1,
    },
    640: {
      slidesPerView: 1,
    },
    768: {
      slidesPerView: 2,
    },
    1024: {
      slidesPerView: 3,
    },
  }
}

function Pages() {
  return (
    <section className="pages section-padding position-re ontop">
      <div className="container-fluid">
        <div className="swiper3">
          <div id="content-carousel-container-unq-clients" className="swiper-container" data-swiper="container">
            <Swiper {...swiperOptions}>
              {data.map(item => (
                <SwiperSlide key={item.id}>
                  <div className="item">
                    <Link href={item.link}>
                      <img src={item.image} alt="" />
                    </Link>
                  </div>
                </SwiperSlide>
              ))}
            </Swiper>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Pages