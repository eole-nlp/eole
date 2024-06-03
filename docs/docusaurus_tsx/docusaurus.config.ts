import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Eole - 👷‍♂️🚧 Work In Progress',
  tagline: 'Open language modeling toolkit based on PyTorch.',
  favicon: 'img/eole-logo.ico',

  // Set the production url of your site here
  url: 'https://eole-nlp.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/eole/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'eole', // Usually your GitHub org/user name.
  projectName: 'eole', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  plugins: [
      // ['@docusaurus/plugin-content-pages', {}],
      ['@docusaurus/plugin-debug', {id: 'debug'}],
      [ require.resolve('docusaurus-lunr-search'), {
        languages: ['en',] // language codes
      }],
      function (context, options) {
        return {
          name: 'webpack-configuration-plugin',
          configureWebpack(config, isServer, utils) {
            return {
              resolve: {
                symlinks: false,
              }
            };
          }
        };
      },
  ],

  markdown: {
    format: "md",

  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/eole-nlp/eole/tree/main/docs',
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/eole-nlp/eole/tree/main/docs',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    // image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: false,
    },
    navbar: {
      title: '',
      logo: {
        alt: 'Eole Logo',
        src: 'img/eole-logo.png',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          type: 'docSidebar',
          sidebarId: 'apiSidebar',
          position: 'left',
          label: 'Reference',
        },
        // {to: '/blog', label: 'Blog', position: 'left'},
        // we'll enable this when we'll start proper versioning
        // {
        //   type: 'docsVersionDropdown',
        //   position: 'right',
        // },
        {
          href: 'https://github.com/eole-nlp/eole',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Docs',
              to: '/docs/',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Github',
              href: 'https://github.com/eole-nlp/eole/discussions',
            },
          ],
        },
        {
          title: 'More',
          items: [
            // {
            //   label: 'Blog',
            //   to: '/blog',
            // },
            {
              label: 'Source',
              href: 'https://github.com/eole-nlp/eole',
            },
          ],
        },
      ],
      // copyright: `Copyright © ${new Date().getFullYear()} My Project, Inc. Built with Docusaurus.`,
      copyright: `EOLE is an open-source toolkit and is licensed under the MIT license.`
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
