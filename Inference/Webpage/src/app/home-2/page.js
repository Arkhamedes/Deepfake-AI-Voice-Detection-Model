import Script from "next/script";
//= Scripts
import generateStylesheetObject from '@/common/generateStylesheetsObject';
//= Common Components
import LoadingScreen from "@/components/Common/Loader";
import MouseCursor from "@/components/Common/MouseCursor";
import ScrollToTop from "@/components/Common/ScrollToTop";
//= Page Components
import TopNavbar from "@/components/CreativeStudio/TopNavbar";
import Header from "@/components/CreativeStudio/Header";
import Services from "@/components/CreativeStudio/Services";
import Featured from "@/components/CreativeStudio/Featured";
import Testimonials from "@/components/CreativeStudio/Testimonials";
import Numbers from "@/components/CreativeStudio/Numbers";
import About from "@/components/CreativeStudio/About";
import Approach from "@/components/CreativeStudio/Approach";
import Blog from "@/components/CreativeStudio/Blog";
import Footer from "@/components/CreativeStudio/Footer";

export const metadata = {
  title: 'Swak - Creative Studio',
  icons: {
    icon: "/home_2/assets/img/fav2.png",
    shortcut: "/home_2/assets/img/fav2.png",
    other: generateStylesheetObject([
      '/common/assets/css/lib/bootstrap.min.css',
      '/common/assets/css/lib/line-awesome.css',
      '/common/assets/css/lib/themify-icons.css',
      '/common/assets/css/lib/ionicons.css',
      '/common/assets/css/lib/animate.css',
      '/common/assets/css/lib/swiper8.min.css',
      '/common/assets/css/lib/cursor.css',
      '/common/assets/css/common_style.css',
      '/home_2/assets/css/home_2_style.css'
    ])
  }
}

export default function Home2() {
  return (
    <body className="home-style2 light-theme">
      <LoadingScreen />
      {/* Page Components */}
      <div className="smooth-scroll-content" id="scrollsmoother-container">
        <TopNavbar />
        <Header />
        <main>
          <Services />
          <Featured />
          <Testimonials />
          <Numbers />
          <About />
          <Approach />
          <Blog />
        </main>
        <Footer />
      </div>
      <MouseCursor />
      <ScrollToTop />
      {/* Page Scripts */}
      <Script src="/common/assets/js/lib/bootstrap.bundle.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/lib/charming.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/lib/TweenMax.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/gsap_lib/gsap.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/gsap_lib/ScrollSmoother.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/gsap_lib/ScrollTrigger.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/gsap_lib/SplitText.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/lib/demo.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/common_js.js" strategy="lazyOnload" />
      <Script src="/home_2/assets/js/home_2_scripts.js" strategy="lazyOnload" />
    </body>
  )
}
