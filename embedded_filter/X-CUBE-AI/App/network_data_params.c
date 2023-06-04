/**
  ******************************************************************************
  * @file    network_data_params.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Sun Jun  4 16:25:07 2023
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "network_data_params.h"


/**  Activations Section  ****************************************************/
ai_handle g_network_activations_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(NULL),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};




/**  Weights Section  ********************************************************/
AI_ALIGNED(32)
const ai_u64 s_network_weights_array_u64[400] = {
  0x3d50b934bf83e0eeU, 0x3efaf8bbc02f907bU, 0x3e883b48c0188445U, 0x3f9cef18bfba5913U,
  0x4036be2ac022cb06U, 0xbffff7143f89bad4U, 0xc024fb98bfafc7bdU, 0x3ec70965404652b6U,
  0x3f9f73e3bfeaa6b2U, 0xbf8892193ee2c496U, 0xbf345945bfea0e18U, 0x4063d83d3fcb5e53U,
  0xbf03ffdac0394edaU, 0x3f4e85c3404c3af5U, 0xbf48976ec00d82c1U, 0xbfb241a6bf0635cdU,
  0xbc3a5bcfbeff52faU, 0x3e14dacebf8dbda4U, 0x3f42bce63f81201aU, 0x3fb87bef3f1bab3aU,
  0x3f529f023f8ac5e5U, 0xbf8597413e95c926U, 0xbd6a7ab3bf381bd9U, 0x3f3b0ac23f891713U,
  0xbf0ea98d4001c93cU, 0x3f35c6493f12b9aaU, 0x3ec5144d3f9ee4f3U, 0x3fa540963dc06558U,
  0x3edf4e0dbe86a867U, 0x3de2edd33f81f674U, 0xbfac09e5bf67a861U, 0x400525563f3eb2c3U,
  0xbf9daef340122a81U, 0x3fc225af3eec2ed6U, 0x3ffbab603fde7dd2U, 0x3fdb117bbfbe7e1fU,
  0xbf78d82f3e0c8cf7U, 0x3f621711bf8f0bfdU, 0x3f17e420bd194b80U, 0x3fbab001bef06405U,
  0xbcbe652f3ef423f4U, 0x3f14f306bebd1a54U, 0x3ed724f8bed1e40aU, 0x3ee922c73e53e9ceU,
  0xbf28e3793db402b8U, 0x3f86fd933ebf80c1U, 0xbdb06b0cbf9a1e5aU, 0x3f901855bf36fba3U,
  0xbe30b26fbe938978U, 0xbe1eab57bf0aceb6U, 0xbd8246adbdaef3e3U, 0xbe2b00b93f4e708dU,
  0x3fc1fa5dbfa16612U, 0xbfb3b9cf3f33e192U, 0xbf3a2cdfbf11a130U, 0x3ed256bf3f4f6b30U,
  0xbe73d3eabf1c2596U, 0x3f04f2b53ed89735U, 0xbeb34898bf23c4d8U, 0xbd035210bf2077c9U,
  0xbef5bed43d6e5af6U, 0x3f1cc336bf1b9c11U, 0x3f5c007e3cc6ef85U, 0xbeb4222bbfaf7708U,
  0x3d9f7e5c3eb1826cU, 0xbe22cf3ebee67b52U, 0x3eb02dd13e739a7bU, 0x3e706f8f3ebbdbffU,
  0xbea070203d420d65U, 0x3eecbfacbea7e23eU, 0x3d048c7a3e96deb1U, 0xbe4891ebbee32c17U,
  0xbfb6ab12bf8012c7U, 0x3f77f38fbff07e20U, 0x3f4f05543f820625U, 0xbe7126fe3ea3518dU,
  0xbe28a27dbee7782cU, 0x3e918fbc3e2200d9U, 0xbe593defbe7b93b1U, 0x3cef4507beac1293U,
  0xbdee7569bc283fa2U, 0x3ec9cdc1be1bcddcU, 0xbdeb432a3f2d9388U, 0xbe8f018fbe8ebd5eU,
  0xbf24f076bf0d7f56U, 0x3fa2c679bf4ec642U, 0x3fa4ebc5be8d10acU, 0xbf8ff4b1bfeb1602U,
  0xbf21d836c0276229U, 0x3ee2f0e43d3a4e3dU, 0xbebf842dbef1d64eU, 0xbe25bb313e17ef3eU,
  0xbf293df83f99e0f4U, 0x3f179f43bf03cc36U, 0x3f87397d3ea3b633U, 0x3f1ec860bf496c0bU,
  0xbeddb8d43ece95edU, 0x3ef174bfbf2bbf8fU, 0xbec62524bf217eabU, 0x402c056c3eac531eU,
  0xbfde04453fb2a678U, 0xbf8d41eabee4619dU, 0xbf09e4a63ec4bf98U, 0x3f5429acbf119764U,
  0x3e45ad9fbeb7bc48U, 0x3c0cb3ec3f591854U, 0x3ee1fbf8bec10fcaU, 0xbf400486beaa2032U,
  0xbef702833e99a6beU, 0x3f2aeb78bf1b0930U, 0x3e3af756bf0df578U, 0x3f92e4a43eaf7967U,
  0x3e3897e3bf349859U, 0x3ee8bc713f2ab11eU, 0x3edac0ae3e02bd06U, 0xbe45cb5abdacaa6dU,
  0xbe9b6ce9bf2a132eU, 0x403b3d0ebcb7c6a2U, 0xbfaf24053f65d5cdU, 0xbf9ed6ddbecb2222U,
  0xbfc78fe1be57cc37U, 0x3edf8c81bf5b6b1dU, 0xbf2dbef8be3d573eU, 0x3f1657ec3f1ee9b2U,
  0x3e2340b6bee2a449U, 0xbe377fa03ea2bf5bU, 0xbf175fb93e31bd7eU, 0x3ec26c2fbfa33708U,
  0x3dcfc482bea531e0U, 0x3f84b6953e05f28dU, 0x3eb7ee29bf2de980U, 0xbe5108563d17aafaU,
  0xbf356904be8dfa70U, 0xbf22de033f0f3aceU, 0xbdecd203bd513f06U, 0x3fbc65bebe71aefdU,
  0xbf88fb193f572a33U, 0xbe88c7edbeb77edcU, 0x3c00c2fc3e16fc8fU, 0xbf33aedfbe57b492U,
  0xbe11115cbed9aa98U, 0x3f20c1983e5d45afU, 0xbf0b10fb3e686489U, 0xbe28b263bec1cb32U,
  0x3eb3da833e4682aeU, 0x3e64bae0be820280U, 0x3b96af79bf34f9aaU, 0x3fb79d323efd081cU,
  0xbd8430f0bf3c42acU, 0xbf23e01e3f87d447U, 0xbf9e7921be2425c3U, 0x3e8e1cf7bf363c8bU,
  0x3f41dec9be739152U, 0xbf2ffebf3f973d6bU, 0xbf2010713f304ff2U, 0xbf22e0a2bebbc8bbU,
  0xbf8b3fcc3e85ee30U, 0x3f8298f7bf2463c1U, 0x3fd4e901beb26c6cU, 0xc01508293f0bd962U,
  0x3fc9eafebf6eaaebU, 0x3e8812bf3e5df4bcU, 0xbf138ee33e61f128U, 0x3e3c88b7bf9ba967U,
  0x3e2b4e97be762c36U, 0x3f231943bd3af7f1U, 0x3f7cc91ebfbad257U, 0x3fc2b5583f839e44U,
  0x0U, 0x0U, 0x0U, 0x0U,
  0x0U, 0x0U, 0x0U, 0x0U,
  0x0U, 0xbdf2daf0bf76f1b0U, 0xbf4e2ff6be930fdeU, 0xbeaf7340bf4fc614U,
  0x3f534dac3f18a2d8U, 0xbd06fab83e97d639U, 0x3e9e08e33c749dc9U, 0xbc2454eebe0db9e6U,
  0xbddd23c73e2eef78U, 0xbc8f33c43d124279U, 0xbdc10c2fbf6a09deU, 0xbf2d7632bf117bc5U,
  0xbefd5c6abf34e96cU, 0xbf9eb8a940154cdaU, 0x3f474b28beace781U, 0xbf0d2839c03df933U,
  0x4012470f3f5fdb33U, 0x3f6cb4983ec6f76aU, 0x3e6a62f03c6dca72U, 0x40c688d23fd06022U,
  0x400317634076551cU, 0xbf160d63c054a9f8U, 0xbfa058064001ca7cU, 0x3fc3ec1f3f8821f7U,
  0x3fc3c2c3406dcaebU, 0xc05642b43f80c501U, 0xbe034513c026cff7U, 0xbf9677fac00e1d3eU,
  0x3fa9cfeac02bcbbcU, 0x3c8dbccebe80b873U, 0xbf0980c6bf8ac200U, 0x408c26223e603acaU,
  0xbeeba00d3f889f5dU, 0xc006366dc061d2d0U, 0xbf5047b63e6c86c5U, 0xbf20e0443e7af440U,
  0x3e57e8ba401c0c8fU, 0x404a462abfc5c54dU, 0x3f2d05f64115ba01U, 0xc02c7d79402205beU,
  0xc0cfe5b53e37103cU, 0xbeac7d3f403d3327U, 0x3fa93706bfaab575U, 0xbf171355bf6b55eeU,
  0xbf7200c4c12387b8U, 0x41157b5f3fa4c352U, 0xc0c008293e84d0f9U, 0xbf1547cac0f31650U,
  0x3f456785c0cea921U, 0x3e8376c3408dd0c1U, 0x40099d0fbf1d728dU, 0xbeaed76bc0284704U,
  0x403ec74cbdfdc640U, 0x40277607bec4e165U, 0xbe328573bf0a5b90U, 0x411bfc073fda1afcU,
  0x40699c184081c482U, 0xbf979f10c0569c04U, 0xbf9e131040145d3aU, 0x3fe96ef83f6b7e2bU,
  0x3faf7c6a409d3b5bU, 0xbe604485be40019dU, 0x3cd82a693e3d1d64U, 0x3e58101fbcda839bU,
  0x3ccaf94fbe3c6807U, 0x3e3f31b13d90fae5U, 0xbd94dda33e21c220U, 0xbf8404c6bf72eb82U,
  0x402095d6c0598a99U, 0x3f7f21f53f77ce63U, 0xc0d3ce57c02025a3U, 0xbe8a6ba2c081dcacU,
  0x3f5ef1d23e3fa7f6U, 0xbfa040f3c007e910U, 0xbf76bdebbfa9ed64U, 0xbe92e25ebfd1853fU,
  0x3dc1ff31bf6291a6U, 0xbf8b43bf3f6104b8U, 0xc01c81e63f515beeU, 0x3f3d650ec07387b1U,
  0x3ef7ad223f7aaffdU, 0xbfedcedec06c2c51U, 0x3d8c317ebf5a22f3U, 0xbe61c572be9e4253U,
  0x3f46953abfeb688eU, 0xbe8cb67b3e382445U, 0xbfc9a3d3bf2d0025U, 0xbe709bf7c03954b0U,
  0x402edd2bbfbc76e5U, 0x3efc549ebfbd3042U, 0xc075acaac07a9b76U, 0x3f9029a0c06667d0U,
  0x3e7c3f7e3fb69584U, 0xbfa80a09c0bf7fe4U, 0x3e0f4bdac03e0a9cU, 0xbfb80da240210068U,
  0x3f904265bf68bed3U, 0x401e3dab3fc9ed36U, 0xbffc190dc01d08d3U, 0x3fa8b1a7c0111476U,
  0xbe987558bf117739U, 0xc01f9677bfa726cfU, 0xbfa62a0a402d82a3U, 0xbe867962bfdeaed4U,
  0xbea976293f101414U, 0xbf98bb29bf66f9f4U, 0x4003ce46400dee44U, 0x3c409810bfaf7644U,
  0x3ec6e904bedc6306U, 0xbeb14e8abe70b596U, 0xbf62d2cdbe83b5c0U, 0x3c85e2cdbfdf5479U,
  0xbd971c5d3e575205U, 0xbfa2b5c1bf8d4b7cU, 0x3fa6cda93f8edc27U, 0xbfc0d3d43fe0a42aU,
  0x402003f3c00be25cU, 0x3f80b3433fde8eb0U, 0xc03cf365c01705f1U, 0x3f3828d2c0002baeU,
  0xbf8554243fd05cf0U, 0xbfad09f5bfe9c25bU, 0x3fa26861401a67d9U, 0x3edf9ebfbec574a8U,
  0xbf3f48753fab1bd0U, 0xbf023a51c045fbf1U, 0x40490adfbde38affU, 0xbf67f65c3fb54387U,
  0x3fcf9aa2bfb9dde8U, 0x3fd10bb13e8a2da9U, 0xbff97eccc0085257U, 0xbe2b8b59bdcd445dU,
  0xbe2d975e3edfc474U, 0x3efd7bcfbf4c7e3fU, 0x3e9f9c08be218da4U, 0xbf94ffc1bfa94850U,
  0x3ca58622bdda3f9bU, 0xbf8512b13cb3bfcfU, 0xbebf8ad83d819c22U, 0x3f77b2f4bfaf9682U,
  0xbe413029c001b2e8U, 0xbf96cc88bf51a21aU, 0xbe560e74bf820926U, 0xbd503c44401075b0U,
  0xbf9eb1883e6d3377U, 0x3f2d84533ec883faU, 0x3f04dcb3bdf8287eU, 0xbf00f4ac3f9d21c8U,
  0xbe42da463f469286U, 0x3f9c4cd3beb0cf2cU, 0x3f3f15d7be747a20U, 0x3e43d1de3f7aba3aU,
  0xbd95d4c03e8e2c3dU, 0x3c15161abf1de548U, 0x3eb57b67bec0e7d7U, 0xbf1f2116bfdd8565U,
  0x3f2f9adebdcb905aU, 0xbea912c43e446ddcU, 0xbd1601dabec8a8b8U, 0x3e3ffe37c047a26bU,
  0x3e909b4abe921aaaU, 0xbf9558bfbf986175U, 0xbee5c1f9bfb18909U, 0x3ff2bc27c0a7453aU,
  0x3d8fb697c0354162U, 0xc085dcbcc0041ea8U, 0xc092e560401aff54U, 0x3e612512c0875978U,
  0xbd2f69733f57789fU, 0xbfdaa88fc0375078U, 0x3ffe417e3f5572a1U, 0x38fe25acc05cb977U,
  0x3ea4af983d24bc67U, 0xbfb9b59b3dcdbab6U, 0xbf9884593eddf242U, 0x3ea0bbadc060c0aeU,
  0xbcad5fa73f563caaU, 0xbff8fce3c0742fa8U, 0x3fe288b63b0f9126U, 0x3e1189c73d2fb6c4U,
  0x3f874dc9bfe8a769U, 0xbec8d3b0bf6783a0U, 0xbf425d08bfabe1d9U, 0x3f61ac04c09774e0U,
  0x3d39db593f1c7798U, 0xc0253231c08b2f78U, 0xbf43b61f3f917839U, 0x3f850d5ec0152616U,
  0xbe46c9f43fdda410U, 0xbf2875d9c0ba9252U, 0x4010f921c007c442U, 0xbfb1e222400b866aU,
  0x3ffe2ac7bf2d5a16U, 0x40175cf83f10d739U, 0xc00bea54c04b6c02U, 0xbc0b97d13e1584dfU,
  0x0U, 0x0U, 0x0U, 0x0U,
  0x0U, 0x0U, 0x0U, 0x0U,
  0x0U, 0x0U, 0x0U, 0x0U,
  0x3dc2cd8b3f75cb3cU, 0xbe8617a13f64d4a5U, 0x3dd71e5a3f3670d8U, 0x3ecde6ccbe4a552bU,
  0x3d8717a23efb7577U, 0xbecdbaddbf867487U, 0x3f224d47bde2c0cdU, 0xbec3bd60bdf422faU,
  0xbe01107f3e8adbf8U, 0x3d3a508dbd0a42a7U, 0x3e26baff3e53c0a1U, 0x3e29031dbc4e05a0U,
  0x3e42b5743fad2f30U, 0xbe856fd23f3fc26dU, 0x3f0c0b983eeed799U, 0x3e6015f7bb88caefU,
};


ai_handle g_network_weights_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(s_network_weights_array_u64),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};

