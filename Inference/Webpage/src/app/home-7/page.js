import Script from "next/script";
//= Scripts
import generateStylesheetObject from '@/common/generateStylesheetsObject';
//= Common Components
import LoadingScreen from "@/components/Common/Loader";
import MouseCursor from "@/components/Common/MouseCursor";
//= Page Components
import SideMenu from "@/components/ConsultingAgency/SideMenu";
import Navbar from "@/components/ConsultingAgency/Navbar";
import Header from "@/components/ConsultingAgency/Header";
import About from "@/components/ConsultingAgency/About";
import Brands from "@/components/ConsultingAgency/Brands";
import Services from "@/components/ConsultingAgency/Services";
import Portfolio from "@/components/ConsultingAgency/Portfolio";
import Testimonials from "@/components/ConsultingAgency/Testimonials";
import Process from "@/components/ConsultingAgency/Process";
import Video from "@/components/ConsultingAgency/Video";
import Blog from "@/components/ConsultingAgency/Blog";
import Footer from "@/components/ConsultingAgency/Footer";

export const metadata = {
  title: 'Swak - Consulting Agency',
  icons: {
    icon: "/home_7/assets/img/fav2.png",
    shortcut: "/home_7/assets/img/fav2.png",
    other: generateStylesheetObject([
      'https://fonts.googleapis.com/css2?family=Prata&display=swap',
      '/common/assets/css/lib/bootstrap.min.css',
      '/common/assets/css/lib/all.min.css',
      '/common/assets/css/lib/themify-icons.css',
      '/common/assets/css/lib/ionicons.css',
      '/common/assets/css/lib/animate.css',
      '/common/assets/css/lib/swiper8.min.css',
      '/common/assets/css/common_style.css',
      '/home_7/assets/css/home_7_style.css'
    ])
  }
}

export default function Home7() {
  return (
    <body className="home-style7">
      <LoadingScreen />
      {/* Page Components */}
      <SideMenu />
      <div id="scrollsmoother-container">
        <Navbar />
        <Header />
        <main>
          <About />
          <Brands />
          <Services />
          <Portfolio />
          <Testimonials />
          <Process />
          <Video />
          <Blog />
        </main>
        <Footer />
      </div>
      <MouseCursor />
      {/* Page Scripts */}
      <Script src="/common/assets/js/lib/bootstrap.bundle.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/gsap_lib/gsap.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/gsap_lib/ScrollSmoother.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/gsap_lib/ScrollTrigger.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/gsap_lib/SplitText.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/common_js.js" strategy="lazyOnload" />
      <Script src="/home_7/assets/js/home_7_scripts.js" strategy="lazyOnload" />
    </body>
  )
}
