import "@/styles/globals.css";
import "@/styles/modal-video.css";
import 'lightgallery/css/lightgallery.css';

export const metadata = {
  title: 'Swak',
  description: 'Swak - Multi-Purpose React.js Next.js Template',
  keywords: ['HTML5', 'Template', 'Swak', 'Multi-Purpose', 'themeforest'],
  icons: {
    icon: "/logo.png",
    shortcut: "/logo.png",
    other: [
      {
        rel: 'stylesheet',
        url: 'https://fonts.googleapis.com/css2?family=Poppins:wght@100;200;300;400;500;600;700;800;900&family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400;1,500;1,600;1,700;1,800&family=Sora:wght@100;200;300;400;500;600;700;800&family=Inter:wght@100;200;300;400;500;600;700;800;900&family=Darker+Grotesque:wght@300;400;500;600;700;800;900&family=Inter:wght@200;300;400;500;700;900&family=Prata&display=swap',
      },
    ]
  },
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      {children}
    </html>
  )
}
