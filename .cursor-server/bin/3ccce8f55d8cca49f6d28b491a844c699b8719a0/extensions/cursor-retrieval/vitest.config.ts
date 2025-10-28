import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
	test: {
		environment: 'node',
		include: ['test/indexing/repoIndexWatcher.test.ts'],
		testTimeout: 30000,
	},
	resolve: {
		alias: {
			'vscode': path.resolve(__dirname, '../test-utils/vscodeMock.ts'),
			'@anysphere/file-service': path.resolve(__dirname, '../test-utils/fileServiceMock.ts'),
			'@cursor/types': path.resolve(__dirname, '../../src/vs/platform/reactivestorage/common/reactiveStorageTypes'),
			'proto': path.resolve(__dirname, '../../src/proto'),
			'test-utils': path.resolve(__dirname, '../test-utils'),
		},
	},
	server: {
		fs: {
			allow: [path.resolve(__dirname, '..')],
		},
	},
});