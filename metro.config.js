/**
 * Metro configuration for React Native
 * https://github.com/facebook/react-native
 *
 * @format
 */

const { getDefaultConfig } = require('metro-config');
const defaultConfig = getDefaultConfig.getDefaultValues(__dirname);
const blacklist = require('metro-config/src/defaults/blacklist');
module.exports = {
  resolver: {
    // react-native-local-resource
    assetExts: [...defaultConfig.resolver.assetExts, 'html', 'bin'],
    blacklistRE: blacklist([/platform_node/]),
  },
  transformer: {
    getTransformOptions: async () => ({
      transform: {
        experimentalImportSupport: false,
        inlineRequires: false,
      },
    }),
  },
};
