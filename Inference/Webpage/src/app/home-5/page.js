import Script from "next/script";
//= Scripts
import generateStylesheetObject from '@/common/generateStylesheetsObject';
//= Common Components
import LoadingScreen from "@/components/Common/Loader";
import MouseCursor from "@/components/Common/MouseCursor";
import ScrollToTop from "@/components/Common/ScrollToTop";
//= Page Components
import Navbar from "@/components/CreativeAgency/Navbar";
import Header from "@/components/CreativeAgency/Header";
import About from "@/components/CreativeAgency/About";
import Services from "@/components/CreativeAgency/Services";
import Portfolio from "@/components/CreativeAgency/Portfolio";
import Awards from "@/components/CreativeAgency/Awards";
import Video from "@/components/CreativeAgency/Video";
import Contact from "@/components/CreativeAgency/Contact";
import Partners from "@/components/CreativeAgency/Partners";
import Team from "@/components/CreativeAgency/Team";
import DiscussWrapper from "@/components/CreativeAgency/DiscussWrapper";
import Testimonials from "@/components/CreativeAgency/Testimonials";
import Blog from "@/components/CreativeAgency/Blog";
import Footer from "@/components/CreativeAgency/Footer";

export const metadata = {
  title: 'Swak - Creative Agency',
  icons: {
    icon: "/home_5/assets/img/fav.png",
    shortcut: "/home_5/assets/img/fav.png",
    other: generateStylesheetObject([
      '/common/assets/css/lib/bootstrap.min.css',
      '/common/assets/css/lib/all.min.css',
      '/common/assets/css/lib/line-awesome.css',
      '/common/assets/css/lib/themify-icons.css',
      '/common/assets/css/lib/ionicons.css',
      '/common/assets/css/lib/animate.css',
      '/common/assets/css/lib/swiper8.min.css',
      '/common/assets/css/common_style.css',
      '/home_5/assets/css/home_5_style.css'
    ])
  }
}

export default function Home5() {
  return (
    <body className="home-style5">
      <LoadingScreen />
      {/* Page Components */}
      <div id="scrollsmoother-container">
        <Navbar />
        <Header />
        <main>
          <About />
          <Services />
          <Portfolio />
          <Awards />
          <Video />
          <Contact />
          <Partners />
          <Team />
          <DiscussWrapper />
          <Testimonials />
          <Blog />
        </main>
        <Footer />
      </div>
      <MouseCursor />
      <ScrollToTop />
      {/* Page Scripts */}
      <Script src="/common/assets/js/lib/bootstrap.bundle.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/gsap_lib/gsap.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/gsap_lib/ScrollSmoother.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/gsap_lib/ScrollTrigger.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/gsap_lib/SplitText.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/common_js.js" strategy="lazyOnload" />
      <Script src="/home_5/assets/js/home_5_scripts.js" strategy="lazyOnload" />
    </body>
  )
}
