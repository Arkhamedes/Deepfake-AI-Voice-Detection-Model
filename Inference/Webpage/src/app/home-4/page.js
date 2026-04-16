import Script from "next/script";
//= Scripts
import generateStylesheetObject from '@/common/generateStylesheetsObject';
//= Common Components
import LoadingScreen from "@/components/Common/Loader";
import MouseCursor from "@/components/Common/MouseCursor";
import ScrollToTop from "@/components/Common/ScrollToTop";
//= Page Components
import SideMenu from "@/components/AppsTechnology/SideMenu";
import Header from "@/components/AppsTechnology/Header";
import Features from "@/components/AppsTechnology/Features";
import Partners from "@/components/AppsTechnology/Partners";
import Works from "@/components/AppsTechnology/Works";
import Newsletter from "@/components/AppsTechnology/Newsletter";
import Screens from "@/components/AppsTechnology/Screens";
import Testimonials from "@/components/AppsTechnology/Testimonials";
import Download from "@/components/AppsTechnology/Download";
import Blog from "@/components/AppsTechnology/Blog";
import Footer from "@/components/AppsTechnology/Footer";

export const metadata = {
  title: 'Swak - Apps Technology',
  icons: {
    icon: "/home_4/assets/img/fav2.png",
    shortcut: "/home_4/assets/img/fav2.png",
    other: generateStylesheetObject([
      'https://fonts.googleapis.com/css2?family=Darker+Grotesque:wght@300;400;500;600;700;800;900&display=swap',
      '/common/assets/css/lib/bootstrap.min.css',
      '/common/assets/css/lib/all.min.css',
      '/common/assets/css/lib/line-awesome.css',
      '/common/assets/css/lib/themify-icons.css',
      '/common/assets/css/lib/ionicons.css',
      '/common/assets/css/lib/animate.css',
      '/common/assets/css/lib/swiper8.min.css',
      '/common/assets/css/common_style.css',
      '/home_4/assets/css/home_4_style.css'
    ])
  }
}

export default function Home4() {
  return (
    <body className="home-style4">
      <LoadingScreen />
      {/* Page Components */}
      <SideMenu />
      <div id="scrollsmoother-container">
        <Header />
        <main>
          <Features />
          <Partners />
          <Works />
          <Newsletter />
          <Screens />
          <Testimonials />
          <Download />
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
      <Script src="/home_4/assets/js/home_4_scripts.js" strategy="lazyOnload" />
    </body>
  )
}
