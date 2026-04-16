import Script from "next/script";
//= Scripts
import generateStylesheetObject from '@/common/generateStylesheetsObject';
//= Common Components
import LoadingScreen from "@/components/Common/Loader";
import MouseCursor from "@/components/Common/MouseCursor";
import ScrollToTop from "@/components/Common/ScrollToTop";
//= Page Components
import SideMenu from "@/components/AgencyPortfolio/SideMenu";
import Navbar from "@/components/AgencyPortfolio/Navbar";
import Header from "@/components/AgencyPortfolio/Header";
import Portfolio from "@/components/AgencyPortfolio/Portfolio";
import Services from "@/components/AgencyPortfolio/Services";
import Partners from "@/components/AgencyPortfolio/Partners";
import Testimonials from "@/components/AgencyPortfolio/Testimonials";
import Team from "@/components/AgencyPortfolio/Team";
import Awards from "@/components/AgencyPortfolio/Awards";
import Footer from "@/components/AgencyPortfolio/Footer";

export const metadata = {
  title: 'Swak - Agency Portfolio',
  icons: {
    icon: "/logo.png",
    shortcut: "/logo.png",
    other: generateStylesheetObject([
      '/common/assets/css/lib/bootstrap.min.css',
      '/common/assets/css/lib/line-awesome.css',
      '/common/assets/css/lib/themify-icons.css',
      '/common/assets/css/lib/ionicons.css',
      '/common/assets/css/lib/animate.css',
      '/common/assets/css/lib/swiper8.min.css',
      '/common/assets/css/lib/cursor.css',
      '/common/assets/css/common_style.css',
      '/home_1/assets/css/home_1_style.css'
    ])
  }
}

export default function Home1() {
  return (
    <body className="home-style1 bg-light1">
      <LoadingScreen />
      {/* Page Components */}
      <SideMenu />
      <div className="smooth-scroll-content" id="scrollsmoother-container">
        <main>
          <div className="main-box-style1 bg-white radius-9 mx-4 my-3 mt-4">
            <Navbar />
            <Header />
            <Portfolio />
          </div>
          <Services />
          <div className="main-box-style1 bg-white radius-9 mx-4 my-3">
            <Partners />
            <Testimonials />
          </div>
          <Team />
          <Awards />
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
      <Script src="/home_1/assets/js/script.js" strategy="lazyOnload" />
    </body>
  )
}
